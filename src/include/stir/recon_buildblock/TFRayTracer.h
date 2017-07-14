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

#include "stir/recon_buildblock/ProjMatrixElemsForOneBinValue.h"
#include "stir/Succeeded.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/framework/tensor.h"

#include <time.h>

#undef ENABLE_POINTGEN

START_NAMESPACE_STIR

class ProjMatrixElemsForOneBin;
class Succeeded;
template <typename elemT> class CartesianCoordinate3D;

class TFRayTracer 
{
  private:
  std::unique_ptr<tensorflow::Session> session;

  // the tensors needed as inputs to the voxel-based raytracing: these act as queues that are filled element-by-element every time that "schedule_point" is called
  tensorflow::Tensor points_in;
  tensorflow::Tensor ray_vec_in;
  tensorflow::Tensor voxel_size_in;
  tensorflow::Tensor norm_const_in;
  tensorflow::TTypes<float, 2>::Tensor points_in_tensor;
  tensorflow::TTypes<float, 2>::Tensor ray_vec_in_tensor;
  tensorflow::TTypes<float, 1>::Tensor voxel_size_in_tensor;
  tensorflow::TTypes<float, 1>::Tensor norm_const_in_tensor;
  
  int chunksize;
  int chunksize_pointgen;
  int cur_pos;

  std::vector<ProjMatrixElemsForOneBinValue> temp_storage;
  void executeInternal();

  public:
  TFRayTracer(int chunksize, int chunksize_pointgen);
  ~TFRayTracer();
  TFRayTracer(const TFRayTracer&) = delete;
  TFRayTracer& operator=(const TFRayTracer&) = delete;

  int getQueueLength();

  void setVoxelSize(CartesianCoordinate3D<float>& voxel_size);

  // puts a new point into place
  void schedulePoint(float px, float py, float pz, float rx, float ry, float rz, float norm_const);

  void schedulePoint(CartesianCoordinate3D<float>& point, CartesianCoordinate3D<float>& ray_vec, float norm_const);

  // acts on all elements at once with the ray tracer and put the result into "retval". Returns the total number of points that have been operated on
  int execute(std::vector<ProjMatrixElemsForOneBinValue>& retval);

#ifdef ENABLE_POINTGEN
  // second functionality that uses TF to generate points along a LOR and then automatically schedules them for ray tracing
  int scheduleLOR(std::vector<CartesianCoordinate3D<float>>& start_point, std::vector<CartesianCoordinate3D<float>>& stop_point, std::vector<CartesianCoordinate3D<float>>& ray_vec, std::vector<float>& norm_const);
#endif
};

END_NAMESPACE_STIR

#endif
