//
//
/*!

  \file
  \ingroup recon_buildblock
  \brief Declaration of stir::RayTraceVoxelsOnCartesianGrid

  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
#ifndef __TFRayTracer_h
#define __TFRayTracer_h

#include "stir/common.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor.h"

#include <time.h>

START_NAMESPACE_STIR

class ProjMatrixElemsForOneBin;
template <typename elemT> class CartesianCoordinate3D;

class TFRayTracer 
{
  private:
  
  std::unique_ptr<tensorflow::Session> session;

  public:
  TFRayTracer();
  ~TFRayTracer();
  TFRayTracer(const TFRayTracer&) = delete;
  TFRayTracer& operator=(const TFRayTracer&) = delete;

  // Method that actually performs the ray tracing
  void 
  RayTraceVoxelsOnCartesianGridTF(ProjMatrixElemsForOneBin& lor, 
				  const CartesianCoordinate3D<float>& start_point, 
				  const CartesianCoordinate3D<float>& end_point, 
				  const CartesianCoordinate3D<float>& voxel_size,
				  const float normalisation_constant = 1.F);
};

END_NAMESPACE_STIR

#endif
