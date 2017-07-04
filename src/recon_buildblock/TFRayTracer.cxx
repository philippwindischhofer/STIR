//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock

  \brief Implementation of RayTraceVoxelsOnCartesianGrid  

  \author Kris Thielemans
  \author Mustapha Sadki
  \author (loosely based on some C code by Matthias Egger)
  \author PARAPET project

*/
/* Modification history:
   KT 30/05/2002 
   start and stop point can now be arbitrarily located
   treatment of LORs parallel to planes is now scale independent (and checked with asserts)
   KT 18/05/2005
   handle LORs in a plane between voxels
*/


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
//ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/TFRaytracer3D-STIR.pb", &def);
  ReadBinaryProto(Env::Default(), "/home/pwindisc/tf-raytracing/TFSiddon3D-STIR.pb", &def);

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

  
  std::cout << start_point.x() << " / " << start_point.y() << " / " << start_point.z() << std::endl;
  std::cout << stop_point.x() << " / " << stop_point.y() << " / " << stop_point.z() << std::endl;
 
  std::cout << voxel_size.x() << " / " << voxel_size.y() << " / " << voxel_size.z() << std::endl;
 
  std::cout << normalisation_constant << std::endl;

  tensorflow::Input::Initializer voxel_dimensions({voxel_size.x(), voxel_size.y(), voxel_size.z()});
  Tensor voxel_dimensions_in = voxel_dimensions.tensor;

  
  tensorflow::Input::Initializer testlor({ {{roundf(start_point.x() + shift_x) * voxel_size.x(), 
	    roundf(start_point.y() + shift_y) * voxel_size.y(), 
	    roundf(start_point.z() + shift_z) * voxel_size.z()}, 
	  {roundf(stop_point.x() + shift_x) * voxel_size.x(), 
	      roundf(stop_point.y() + shift_y) * voxel_size.y(), 
	      roundf(stop_point.z() + shift_z) * voxel_size.z() }} });

  //std::cout << roundf(start_point.x() + shift_x) * voxel_size.x() << " / " << roundf(start_point.y() + shift_y) * voxel_size.y() << " / " << roundf(start_point.z() + shift_z) * voxel_size.z() << " // " << roundf(stop_point.x() + shift_x) * voxel_size.x() << " / " << roundf(stop_point.y() + shift_y) * voxel_size.y() << " / " << roundf(stop_point.z() + shift_z) * voxel_size.z() << std::endl;

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
    // format: (intersection length, x, y, z)
    if((lengtharr(ii, 0) != 0.f) && (!isnan(lengtharr(ii, 0))))
    {
      CartesianCoordinate3D<int> cur_voxel(lengtharr(ii, 3) - shift_z, lengtharr(ii, 2) - shift_y, lengtharr(ii, 1) - shift_x);
      float cur_val = lengtharr(ii, 0);
      lor.push_back(ProjMatrixElemsForOneBin::value_type(cur_voxel, cur_val));

      std::cout << cur_voxel.x() << " / " << cur_voxel.y() << " / " << cur_voxel.z() << " -- " << cur_val << std::endl;
    }
  }
  
  /*
  // now go through the output length array, extract the nonzero entries and append them to the STIR object
  for(int ii = 0; ii < lengtharr.dimension(0); ii++)
  {
    for(int jj = 0; jj < lengtharr.dimension(1); jj++)
    {
      for(int kk = 0; kk < lengtharr.dimension(2); kk++)
      {
	if(lengtharr(ii, jj, kk) != 0.0f)
	{
	  //std::cout << ii  << " / " << jj << " / " << kk << " - " << lengtharr(ii, jj, kk) << std::endl;
	  CartesianCoordinate3D<int> cur_voxel(kk - shift_z, jj - shift_y, ii - shift_x);
	  CartesianCoordinate3D<int> empty_voxel(0,0,-1);
	  float cur_val = lengtharr(ii, jj, kk);

	  //std::cout << cur_voxel.x() << " / " << cur_voxel.y() << " / " << cur_voxel.z() << " -- " << cur_val << std::endl;

	  lor.push_back(ProjMatrixElemsForOneBin::value_type(cur_voxel, cur_val));
	  //lor.push_back(ProjMatrixElemsForOneBin::value_type(empty_voxel, 10.0f));
	}
      }
    }
  }
  */

 // std::cout << "end raytracer" << std::endl;
}
END_NAMESPACE_STIR
