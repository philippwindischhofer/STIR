#include "stir/recon_buildblock/TFRayTracer.h"

#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/round.h"
#include <math.h>
#include <algorithm>

START_NAMESPACE_STIR

using namespace tensorflow;
using namespace ::tensorflow::ops;

// default constructor, with standard tensorflow session options
TFRayTracer::TFRayTracer(int chunksize, int chunksize_pointgen) : session(NewSession(tensorflow::SessionOptions({}))), 
					  points_in(DT_FLOAT, TensorShape({chunksize, 3})), ray_vec_in(DT_FLOAT, TensorShape({chunksize, 3})), 
					  voxel_size_in(DT_FLOAT, TensorShape({3})), norm_const_in(DT_FLOAT, TensorShape({chunksize})), 
					  points_in_tensor(points_in.tensor<float, 2>()), ray_vec_in_tensor(ray_vec_in.tensor<float, 2>()), 
					  voxel_size_in_tensor(voxel_size_in.tensor<float, 1>()), norm_const_in_tensor(norm_const_in.tensor<float, 1>()), 
  chunksize(chunksize), chunksize_pointgen(chunksize_pointgen), cur_pos(0)
{
  std::cout << "new ray tracer default constructor\n";
  srand(time(NULL));

  // put here all tensors to zero, or their default values (such that even if they have not been completely filled by the user, they are zero-padded)
  points_in_tensor.setZero();
  ray_vec_in_tensor.setZero();
  norm_const_in_tensor.setZero();
  voxel_size_in_tensor.setZero();

  temp_storage.clear();

  Scope root = Scope::NewRootScope();

  GraphDef def;

  // load the correct TF-Graph back from the Protobuf file
#ifdef ENABLE_POINTGEN
  ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/iterative-python/TFRayMarchingWithLORGen.pb", &def);
#else
  ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/iterative-python/TFRayMarching.pb", &def);
#endif

  TF_CHECK_OK(session -> Create(def));
}

TFRayTracer::~TFRayTracer()
{
  TF_CHECK_OK(session -> Close());
}

int TFRayTracer::getQueueLength()
{
  return cur_pos;
}

 void TFRayTracer::setVoxelSize(CartesianCoordinate3D<float>& voxel_size)
 {
   voxel_size_in_tensor(0) = voxel_size.x(); 
   voxel_size_in_tensor(1) = voxel_size.y();
   voxel_size_in_tensor(2) = voxel_size.z();
 }

void TFRayTracer::schedulePoint(CartesianCoordinate3D<float>& point, CartesianCoordinate3D<float>& ray_vec, float norm_const)
{
  schedulePoint(point.x(), point.y(), point.z(), ray_vec.x(), ray_vec.y(), ray_vec.z(), norm_const);
}

void TFRayTracer::schedulePoint(float px, float py, float pz, float rx, float ry, float rz, float norm_const)
{
  // put the new point and its auxiliary information into the input tensors, if there is still free space
  if(cur_pos < chunksize)
  {
    points_in_tensor(cur_pos, 0) = px;
    points_in_tensor(cur_pos, 1) = py;
    points_in_tensor(cur_pos, 2) = pz;

    ray_vec_in_tensor(cur_pos, 0) = rx;
    ray_vec_in_tensor(cur_pos, 1) = ry;
    ray_vec_in_tensor(cur_pos, 2) = rz;

    norm_const_in_tensor(cur_pos) = norm_const;

    cur_pos++;
  }
  else
  {
    // input queue is full, have to execute the points scheduled so far, and store them in the private output queue
    executeInternal();
  }
}

void TFRayTracer::executeInternal()
{
   std::vector<Tensor> outputs;

   // collects all input parameters to the ray marcher
   std::vector<std::pair<string, Tensor>> inputs = {
     {"points", points_in},
     {"ray_vec", ray_vec_in},
     {"voxel_size", voxel_size_in},
     {"norm_const", norm_const_in}
   };

   auto started = std::chrono::high_resolution_clock::now();
   TF_CHECK_OK(session -> Run(inputs, {"out"}, {}, &outputs));
   auto done = std::chrono::high_resolution_clock::now();
   std::cout << "GPU=" << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << std::endl;	 

   // this contains the results of the ray marching for individual voxels in the following format
   // LOI | voxel index x | voxel index y | voxel index z
   auto rt_result = outputs[0].tensor<float, 2>();

   auto started_copying = std::chrono::high_resolution_clock::now();

   // go through the list and put it into the actual return value structure. Must get back exactly as many results as were put in, prescribed by "cur_pos"!
   for(int ii = 0; ii < cur_pos; ii++)
     {
       CartesianCoordinate3D<int> cur_voxel(rt_result(ii, 3), rt_result(ii, 2), rt_result(ii, 1));
       float cur_val = rt_result(ii, 0);

       temp_storage.push_back(ProjMatrixElemsForOneBin::value_type(cur_voxel, cur_val));	 
     }   

   auto done_copying = std::chrono::high_resolution_clock::now();
   std::cout << "copyLL=" << std::chrono::duration_cast<std::chrono::milliseconds>(done_copying - started_copying).count() << std::endl;	 
   // reset the position for the input queue
   cur_pos = 0;
}

 int TFRayTracer::execute(std::vector<ProjMatrixElemsForOneBinValue>& retval)
{
  if(getQueueLength() == 0)
    return 0;

  // now the input buffer may not be full. So, run first the internal execution, and then return the contents of the internal storage and the total number of treated voxels
  executeInternal();
  int number_traced = temp_storage.size();

  // append the contents of the temporary storage to the contents of retval (do not overwrite anything)
  retval.insert(std::end(retval), std::begin(temp_storage), std::end(temp_storage));

  temp_storage.clear();
  
  return number_traced;
}

#ifdef ENABLE_POINTGEN
int TFRayTracer::scheduleLOR(std::vector<CartesianCoordinate3D<float>>& start_point, std::vector<CartesianCoordinate3D<float>>& stop_point, std::vector<CartesianCoordinate3D<float>>& ray_vec, std::vector<float>& norm_const)
{
  int number_lors = start_point.size();
  tensorflow::Tensor start_point_in(DT_FLOAT, TensorShape({number_lors, 3}));
  tensorflow::Tensor stop_point_in(DT_FLOAT, TensorShape({number_lors, 3}));

  auto start_point_in_tensor = start_point_in.tensor<float, 2>();
  auto stop_point_in_tensor = stop_point_in.tensor<float, 2>();

  // fill the input tensors here
  for(int ii = 0; ii < number_lors; ii++)
    {
      start_point_in_tensor(ii, 0) = start_point[ii].x();
      start_point_in_tensor(ii, 1) = start_point[ii].y();
      start_point_in_tensor(ii, 2) = start_point[ii].z();

      stop_point_in_tensor(ii, 0) = stop_point[ii].x();
      stop_point_in_tensor(ii, 1) = stop_point[ii].y();
      stop_point_in_tensor(ii, 2) = stop_point[ii].z();

      norm_const_in_tensor(ii) = norm_const[ii];
    }

  std::vector<Tensor> outputs;
  std::vector<std::pair<string, Tensor>> inputs = {
     {"start_points", start_point_in},
     {"stop_points", stop_point_in},
   };

  TF_CHECK_OK(session -> Run(inputs, {"outpoints"}, {}, &outputs));
  auto rt_result = outputs[0].tensor<float, 2>();

  // print the output tensor (later: also put the computation of the ray vec here and fill directly into the other input queue for ray tracing)
  for(int ii = 0; ii < rt_result.dimension(0); ii++)
    {
      int ray_vec_index = int(ii / chunksize_pointgen);
      schedulePoint(rt_result(ii, 0), rt_result(ii, 1), rt_result(ii, 2), ray_vec[ray_vec_index].x(), ray_vec[ray_vec_index].y(), ray_vec[ray_vec_index].z(), norm_const[ray_vec_index]);
    }

  return rt_result.dimension(0); // number of points that have been added
}
#endif

END_NAMESPACE_STIR
