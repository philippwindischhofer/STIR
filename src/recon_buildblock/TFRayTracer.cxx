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
TFRayTracer::TFRayTracer(int chunksize) : session(NewSession(tensorflow::SessionOptions({}))), 
					  points_in(DT_FLOAT, TensorShape({chunksize, 3})), ray_vec_in(DT_FLOAT, TensorShape({chunksize, 3})), 
					  voxel_size_in(DT_FLOAT, TensorShape({3})), norm_const_in(DT_FLOAT, TensorShape({chunksize})), 
					  points_in_tensor(points_in.tensor<float, 2>()), ray_vec_in_tensor(ray_vec_in.tensor<float, 2>()), 
					  voxel_size_in_tensor(voxel_size_in.tensor<float, 1>()), norm_const_in_tensor(norm_const_in.tensor<float, 1>()), 
  chunksize(chunksize), cur_pos(0)
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
  //ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/TFSiddon3D-STIR.pb", &def);
  //ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/iterative-python/TFRaytracingIterative-STIR.pb", &def);

  // load the correct TF-Graph back from the Protobuf file
  ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/iterative-python/TFRayMarching.pb", &def);

  TF_CHECK_OK(session -> Create(def));
}

TFRayTracer::~TFRayTracer()
{
  TF_CHECK_OK(session -> Close());
}

 void TFRayTracer::setVoxelSize(CartesianCoordinate3D<float>& voxel_size)
 {
   voxel_size_in_tensor(0) = voxel_size.x(); 
   voxel_size_in_tensor(1) = voxel_size.y();
   voxel_size_in_tensor(2) = voxel_size.z();
 }

 void TFRayTracer::schedulePoint(CartesianCoordinate3D<float>& point, CartesianCoordinate3D<float>& ray_vec, float norm_const)
{
  // put the new point and its auxiliary information into the input tensors, if there is still free space
  if(cur_pos < chunksize)
  {
    points_in_tensor(cur_pos, 0) = point.x();
    points_in_tensor(cur_pos, 1) = point.y();
    points_in_tensor(cur_pos, 2) = point.z();

    ray_vec_in_tensor(cur_pos, 0) = ray_vec.x();
    ray_vec_in_tensor(cur_pos, 1) = ray_vec.y();
    ray_vec_in_tensor(cur_pos, 2) = ray_vec.z();

    norm_const_in_tensor(cur_pos) = norm_const;

    cur_pos++;
  }
  else
  {
    // input queue is full, have to execute the points scheduled so far, and store them in the private output queue
    // std::cout << "execute internal was called" << std::endl;
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

   /*
   std::cout << "starting to execute" << std::endl;
   std::cout << points_in_tensor << std::endl;
   std::cout << ray_vec_in_tensor << std::endl;
   std::cout << voxel_size_in_tensor << std::endl;
   std::cout << norm_const_in_tensor << std::endl;
   */

   TF_CHECK_OK(session -> Run(inputs, {"out"}, {}, &outputs));

   // this contains the results of the ray marching for individual voxels in the following format
   // LOI | voxel index x | voxel index y | voxel index z
   auto rt_result = outputs[0].tensor<float, 2>();

   // go through the list and put it into the actual return value structure. Must get back exactly as many results as were put in, prescribed by "cur_pos"!
   for(int ii = 0; ii < cur_pos; ii++)
     {
       CartesianCoordinate3D<int> cur_voxel(rt_result(ii, 3), rt_result(ii, 2), rt_result(ii, 1));
       float cur_val = rt_result(ii, 0);
      
       temp_storage.push_back(ProjMatrixElemsForOneBin::value_type(cur_voxel, cur_val));

       // std::cout << cur_voxel.x() << " / " << cur_voxel.y() << " / " << cur_voxel.z() << " -- " << cur_val << std::endl;
	 
     }   

   // reset the position for the input queue
   cur_pos = 0;
}

 int TFRayTracer::execute(std::vector<ProjMatrixElemsForOneBinValue>& retval)
{
   // now the input buffer may not be full. So, run first the internal execution, and then return the contents of the internal storage and the total number of treated voxels
   executeInternal();
   int number_traced = temp_storage.size();

   // append the contents of the temporary storage to the contents of retval (do not overwrite anything)
   retval.insert(std::end(retval), std::begin(temp_storage), std::end(temp_storage));

   temp_storage.clear();
  
   return number_traced;
}

END_NAMESPACE_STIR
