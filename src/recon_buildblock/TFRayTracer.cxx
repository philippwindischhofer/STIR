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

 Succeeded TFRayTracer::schedulePoint(CartesianCoordinate3D<float>& point, CartesianCoordinate3D<float>& ray_vec, float norm_const)
{
  Succeeded retval = Succeeded::no;

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
    retval = Succeeded::yes;
  }

   return retval;
}

 int TFRayTracer::execute(std::vector<ProjMatrixElemsForOneBinValue>& retval)
{
   std::vector<Tensor> outputs;
   int number_traced = cur_pos;

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

   // go through the list and put it into the actual return value structure
   for(int ii = 0; ii < rt_result.dimension(0); ii++)
     {

       CartesianCoordinate3D<int> cur_voxel(rt_result(ii, 3), rt_result(ii, 2), rt_result(ii, 1));
       float cur_val = rt_result(ii, 0);
      
       retval.push_back(ProjMatrixElemsForOneBin::value_type(cur_voxel, cur_val));

       //std::cout << cur_voxel.x() << " / " << cur_voxel.y() << " / " << cur_voxel.z() << " -- " << cur_val << std::endl;
	 
     }   

   cur_pos = 0;
   return number_traced;
}

// legacy function that traces an explicitely given LOR -> will be discontinued soon
void TFRayTracer::RayTraceVoxelsOnCartesianGridTF
        (ProjMatrixElemsForOneBin& lor, 
         const CartesianCoordinate3D<float>& start_point, 
         const CartesianCoordinate3D<float>& stop_point, 
         const CartesianCoordinate3D<float>& voxel_size,
         const float normalisation_constant)
{

  const float shift_x = 100.f;
  const float shift_y = 100.f;
  const float shift_z = 50.;

  /*
  std::cout << start_point.x() << " / " << start_point.y() << " / " << start_point.z() << std::endl;
  std::cout << stop_point.x() << " / " << stop_point.y() << " / " << stop_point.z() << std::endl;
 
  std::cout << voxel_size.x() << " / " << voxel_size.y() << " / " << voxel_size.z() << std::endl;
 
  std::cout << normalisation_constant << std::endl;
  */

  tensorflow::Input::Initializer voxel_dimensions({voxel_size.x(), voxel_size.y(), voxel_size.z()});
  Tensor voxel_dimensions_in = voxel_dimensions.tensor;

  // convert the voxel coordinates into actual, physically meaningful numbers
  tensorflow::Input::Initializer testlor({ {{(start_point.x() + shift_x) * voxel_size.x(), 
	    (start_point.y() + shift_y) * voxel_size.y(), 
	    (start_point.z() + shift_z) * voxel_size.z()}, 
	  {(stop_point.x() + shift_x) * voxel_size.x(), 
	      (stop_point.y() + shift_y) * voxel_size.y(), 
	      (stop_point.z() + shift_z) * voxel_size.z() }} });

  //std::cout << (start_point.x() + shift_x) * voxel_size.x() << " / " << (start_point.y() + shift_y) * voxel_size.y() << " / " << (start_point.z() + shift_z) * voxel_size.z() << " // " << (stop_point.x() + shift_x) * voxel_size.x() << " / " << (stop_point.y() + shift_y) * voxel_size.y() << " / " << (stop_point.z() + shift_z) * voxel_size.z() << std::endl;

  Tensor testlor_in = testlor.tensor;
  // std::cout << testlor_in.tensor<float,3>() << std::endl;
  // std::cout << voxel_dimensions_in.tensor<float,1>() << std::endl;

  tensorflow::Input::Initializer norm_const({normalisation_constant});
  Tensor norm_const_in = norm_const.tensor;

  //Tensor testlor_in = build_lor(start_point, stop_point, shift_x, shift_y, shift_z);

  std::vector<Tensor> outputs;

  std::vector<std::pair<string, Tensor>> inputs = {
    {"lor", testlor_in}, 
    {"voxel_dims", voxel_dimensions_in},
    {"norm_const", norm_const_in}
  };

  TF_CHECK_OK(session -> Run(inputs, {"la"}, {}, &outputs));
  auto lengtharr = outputs[0].tensor<float, 2>();
  
  // now go through the returned array and extract the coordinates of the voxels and the intersection lengths
  for(int ii = 0; ii < lengtharr.dimension(0); ii++)
  {
    //std::cout << std::isnormal(lengtharr(ii, 0)) << std::endl;
    // format: (intersection length, x, y, z)
    if(std::isnormal(lengtharr(ii, 0)))
    {
      CartesianCoordinate3D<int> cur_voxel(lengtharr(ii, 3) - shift_z, lengtharr(ii, 2) - shift_y, lengtharr(ii, 1) - shift_x);
      float cur_val = lengtharr(ii, 0);
      
      lor.push_back(ProjMatrixElemsForOneBin::value_type(cur_voxel, cur_val));

      //std::cout << cur_voxel.x() << " / " << cur_voxel.y() << " / " << cur_voxel.z() << " -- " << cur_val << std::endl;

    }
  }

 // std::cout << "end raytracer" << std::endl;
}
END_NAMESPACE_STIR
