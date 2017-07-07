#include "stir/recon_buildblock/TFRayTracer.h"

#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/round.h"
#include <math.h>
#include <algorithm>

#ifndef STIR_NO_NAMESPACE
using std::min;
using std::max;
#endif

START_NAMESPACE_STIR

using namespace tensorflow;
using namespace ::tensorflow::ops;

static inline bool
is_half_integer(const float a)
{
  return
    fabs(floor(a)+.5F - a)<.0001F;
}

// not needed at the moment, since graph is read from file
GraphDef createGraph(Scope root)
{
  // construct the graph here
  auto r = Placeholder(root.WithOpName("in"), DT_FLOAT);
  auto c = Add(root.WithOpName("rp"), r, Const(root, 1.0f));

  // convert it to an honest GraphDef object
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));

  return(def);
}

// default constructor, with standard tensorflow session options
TFRayTracer::TFRayTracer() : session(NewSession(tensorflow::SessionOptions({})))
{
  std::cout << "new ray tracer default constructor\n";
  srand(time(NULL));

  Scope root = Scope::NewRootScope();

  GraphDef def;
  //ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/TFSiddon3D-STIR.pb", &def);
  ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/iterative-python/TFRaytracingIterative-STIR.pb", &def);

  TF_CHECK_OK(session -> Create(def));
}

TFRayTracer::~TFRayTracer()
{
  TF_CHECK_OK(session -> Close());
}

// takes care of the shifted coordinate origin etc.
Tensor build_lor(const CartesianCoordinate3D<float>& start_point, const CartesianCoordinate3D<float>& stop_point, const float shift_x, const float shift_y, const float shift_z)
{
  // TODO: maybe put random numbers here?
  const float eps_x = 0.55 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  const float eps_y = 0.55 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  const float eps_z = 0.55 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

  tensorflow::Input::Initializer testlor({ {{roundf(start_point.x() + shift_x), 
	    roundf(start_point.y() + shift_y), 
	    roundf(start_point.z() + shift_z)}, 
	  {roundf(stop_point.x() + shift_x) + eps_x, 
	      roundf(stop_point.y() + shift_y) + eps_y, 
	      roundf(stop_point.z() + shift_z) + eps_z }} });
 
  Tensor testlor_in = testlor.tensor;
  std::cout << testlor_in.tensor<float,3>() << std::endl;

  return testlor_in;
}

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
