//
//
/*!

  \file
  \ingroup projection

  \brief implementations for stir::ForwardProjectorByBinUsingProjMatrixByBin 
   
  \author Kris Thielemans
  \author PARAPET project

*/
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


#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/Viewgram.h"
#include "stir/RelatedViewgrams.h"
#include "stir/IndexRange2D.h"
#include "stir/is_null_ptr.h"
#include <algorithm>
#include <vector>
#include <list>

#include <chrono>

#ifndef STIR_NO_NAMESPACE
using std::find;
using std::vector;
using std::list;
using std::auto_ptr;
#endif

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char * const 
ForwardProjectorByBinUsingProjMatrixByBin::registered_name =
  "Matrix";


void
ForwardProjectorByBinUsingProjMatrixByBin::
set_defaults()
{
  this->proj_matrix_ptr.reset();
  //ForwardProjectorByBin::set_defaults();
}

void
ForwardProjectorByBinUsingProjMatrixByBin::
initialise_keymap()
{
  std::cout << "this is setting up the parameter parsing\n";
  parser.add_start_key("Forward Projector Using Matrix Parameters");
  parser.add_stop_key("End Forward Projector Using Matrix Parameters");
  parser.add_parsing_key("matrix type", &proj_matrix_ptr);
  //ForwardProjectorByBin::initialise_keymap();
}

bool
ForwardProjectorByBinUsingProjMatrixByBin::
post_processing()
{
  //if (ForwardProjectorByBin::post_processing() == true)
  //  return true;
  if (is_null_ptr(proj_matrix_ptr))
  { 
    warning("ForwardProjectorByBinUsingProjMatrixByBin: matrix not set.\n");
    return true;
  }
  return false;
}

ForwardProjectorByBinUsingProjMatrixByBin::
ForwardProjectorByBinUsingProjMatrixByBin()
{
  std::cout << "this is FordwardProjectorByBinUsingProjMatrixByBin default constructor\n";
  set_defaults();
}

ForwardProjectorByBinUsingProjMatrixByBin::
ForwardProjectorByBinUsingProjMatrixByBin(  
    const shared_ptr<ProjMatrixByBin>& proj_matrix_ptr
    )
  : proj_matrix_ptr(proj_matrix_ptr)
{
  assert(!is_null_ptr(proj_matrix_ptr));
}

void
ForwardProjectorByBinUsingProjMatrixByBin::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)
{    	   
  proj_matrix_ptr->set_up(proj_data_info_ptr, image_info_ptr);
  
  std::string infostring = proj_matrix_ptr -> get_registered_name();
  if(infostring == ProjMatrixByBinUsingRayTracingTF::registered_name)
    TF_enabled = true;
  else
    TF_enabled = false;
}

const DataSymmetriesForViewSegmentNumbers *
ForwardProjectorByBinUsingProjMatrixByBin::get_symmetries_used() const
{
  return proj_matrix_ptr->get_symmetries_ptr();
}

void 
ForwardProjectorByBinUsingProjMatrixByBin::
 actual_forward_project(RelatedViewgrams<float>& viewgrams, 
		  const DiscretisedDensity<3,float>& image,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num)
{

  auto started_actual_fw = std::chrono::high_resolution_clock::now();

  CPUTimer timer;
  timer.reset();

  std::cout << "-- forward_project_by_bin actual\n";

  if(TF_enabled)
    {
      std::cout << "TF available in forward projector" << std::endl;

      // this is the TF-enabled version. to use the new features, need to downcast the pointer first!
      shared_ptr<ProjMatrixByBinUsingRayTracingTF> proj_matrix_ptr_tf = boost::dynamic_pointer_cast<ProjMatrixByBinUsingRayTracingTF> (proj_matrix_ptr);
    
      //ProjMatrixElemsForOneBin proj_matrix_row;
      std::vector<ProjMatrixElemsForOneBin> proj_matrix_rows;
      RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams.begin();
    
      while( r_viewgrams_iter!=viewgrams.end())
	{
	  int matrix_element_cnt = 0;

	  Viewgram<float>& viewgram = *r_viewgrams_iter;
	  const int view_num = viewgram.get_view_num();
	  const int segment_num = viewgram.get_segment_num();
	  
	  auto started_scheduling = std::chrono::high_resolution_clock::now();
      
	  for ( int tang_pos = min_tangential_pos_num ;tang_pos  <= max_tangential_pos_num ;++tang_pos)  
	    {

	      for ( int ax_pos = min_axial_pos_num; ax_pos <= max_axial_pos_num ;++ax_pos)
		{ 
		  Bin bin(segment_num, view_num, ax_pos, tang_pos, 0);
		  proj_matrix_ptr_tf -> schedule_matrix_elems_for_one_bin(bin);
		  matrix_element_cnt++;
		}
	    }

	  auto started_raytracing = std::chrono::high_resolution_clock::now();
	  proj_matrix_rows.clear();
	  proj_matrix_ptr_tf -> execute(proj_matrix_rows);
	  
	  auto started_fwprojection = std::chrono::high_resolution_clock::now();

	  int element_cnt = 0;
	  for ( int tang_pos = min_tangential_pos_num ;tang_pos  <= max_tangential_pos_num ;++tang_pos)  
	  {
	      for ( int ax_pos = min_axial_pos_num; ax_pos <= max_axial_pos_num ;++ax_pos)
		{ 
		  Bin bin(segment_num, view_num, ax_pos, tang_pos, 0);
		  
		  proj_matrix_rows[element_cnt].forward_project(bin, image);		  
		  viewgram[ax_pos][tang_pos] = bin.get_bin_value();    
		  element_cnt++;
		}
	    }
	   
	  auto done = std::chrono::high_resolution_clock::now();

	  
	  std::cout << "scheduling=" << std::chrono::duration_cast<std::chrono::milliseconds>(started_raytracing - started_scheduling).count() << std::endl;
	  std::cout << "ray_tracing=" << std::chrono::duration_cast<std::chrono::milliseconds>(started_fwprojection - started_raytracing).count() << std::endl;	 
	  std::cout << "forwardprojection=" << std::chrono::duration_cast<std::chrono::milliseconds>(done - started_fwprojection).count() << std::endl;
	  std::cout << "total=" << std::chrono::duration_cast<std::chrono::milliseconds>(done - started_scheduling).count() << std::endl;
	  std::cout << "matrix_element_count=" << matrix_element_cnt << std::endl;
	  
	  
	  ++r_viewgrams_iter; 	  
	}	   
    }
  else
    {
      // otherwise, continue in legacy mode
      if (proj_matrix_ptr->is_cache_enabled()/* &&
						!proj_matrix_ptr->does_cache_store_only_basic_bins()*/)
	{
	  // straightforward version which relies on ProjMatrixByBin to sort out all 
	  // symmetries
	  // would be slow if there's no caching at all, but is very fast if everything is cached
    
	  ProjMatrixElemsForOneBin proj_matrix_row;
    
	  RelatedViewgrams<float>::iterator r_viewgrams_iter = viewgrams.begin();
    
	  while( r_viewgrams_iter!=viewgrams.end())
	    {
	      auto started = std::chrono::high_resolution_clock::now();

	      Viewgram<float>& viewgram = *r_viewgrams_iter;
	      const int view_num = viewgram.get_view_num();
	      const int segment_num = viewgram.get_segment_num();
      
	      std::cout << "start_matrix_chunk" << std::endl;

	      for ( int tang_pos = min_tangential_pos_num ;tang_pos  <= max_tangential_pos_num ;++tang_pos)  
		{
		  for ( int ax_pos = min_axial_pos_num; ax_pos <= max_axial_pos_num ;++ax_pos)
		    { 
		      Bin bin(segment_num, view_num, ax_pos, tang_pos, 0);

		      proj_matrix_ptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, bin);
		    		    
		      proj_matrix_row.forward_project(bin,image);
		      viewgram[ax_pos][tang_pos] = bin.get_bin_value();
		    }
		}

	      std::cout << "end_matrix_chunk" << std::endl;

	      auto done = std::chrono::high_resolution_clock::now();

	      std::cout << "total=" << std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count() << std::endl;

	      ++r_viewgrams_iter; 
	    }	   
	}
      else
	{
	  // Complicated version which handles the symmetries explicitly.
	  // Faster when no caching is performed, about just as fast when there is caching, 
	  // but of only basic bins.
    
	  ProjMatrixElemsForOneBin proj_matrix_row;
	  ProjMatrixElemsForOneBin proj_matrix_row_copy;
	  const DataSymmetriesForBins* symmetries = proj_matrix_ptr->get_symmetries_ptr(); 
    
	  Array<2,int> 
	    already_processed(IndexRange2D(min_axial_pos_num, max_axial_pos_num,
					   min_tangential_pos_num, max_tangential_pos_num));
    
	  for ( int tang_pos = min_tangential_pos_num ;tang_pos  <= max_tangential_pos_num ;++tang_pos)  
	    for ( int ax_pos = min_axial_pos_num; ax_pos <= max_axial_pos_num ;++ax_pos)
	      {       
		if (already_processed[ax_pos][tang_pos])
		  continue;          
        
		Bin basic_bin(viewgrams.get_basic_segment_num(),viewgrams.get_basic_view_num(),ax_pos,tang_pos);
		symmetries->find_basic_bin(basic_bin);
        
		proj_matrix_ptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, basic_bin);
        
		vector<AxTangPosNumbers> r_ax_poss;
		symmetries->get_related_bins_factorised(r_ax_poss,basic_bin,
							min_axial_pos_num, max_axial_pos_num,
							min_tangential_pos_num, max_tangential_pos_num);
        
		for (
#ifndef STIR_NO_NAMESPACES
		     std::
#endif
		       vector<AxTangPosNumbers>::iterator r_ax_poss_iter = r_ax_poss.begin();
		     r_ax_poss_iter != r_ax_poss.end();
		     ++r_ax_poss_iter)
		  {
		    const int axial_pos_tmp = (*r_ax_poss_iter)[1];
		    const int tang_pos_tmp = (*r_ax_poss_iter)[2];
          
		    // symmetries might take the ranges out of what the user wants
		    if ( !(min_axial_pos_num <= axial_pos_tmp && axial_pos_tmp <= max_axial_pos_num &&
			   min_tangential_pos_num <=tang_pos_tmp  && tang_pos_tmp <= max_tangential_pos_num))
		      continue;
          
		    already_processed[axial_pos_tmp][tang_pos_tmp] = 1;
          
          
		    for (RelatedViewgrams<float>::iterator viewgram_iter = viewgrams.begin();
			 viewgram_iter != viewgrams.end();
			 ++viewgram_iter)
		      {
			Viewgram<float>& viewgram = *viewgram_iter;
			proj_matrix_row_copy = proj_matrix_row;
			Bin bin(viewgram_iter->get_segment_num(),
				viewgram_iter->get_view_num(),
				axial_pos_tmp,
				tang_pos_tmp);
            
			auto_ptr<SymmetryOperation> symm_op_ptr = 
			  symmetries->find_symmetry_operation_from_basic_bin(bin);
			assert(bin == basic_bin);
            
			symm_op_ptr->transform_proj_matrix_elems_for_one_bin(proj_matrix_row_copy);
			proj_matrix_row_copy.forward_project(bin,image);
            
			viewgram[axial_pos_tmp][tang_pos_tmp] = bin.get_bin_value();
		      }
		  }  
	      }      
	  assert(already_processed.sum() == (
					     (max_axial_pos_num - min_axial_pos_num + 1) *
					     (max_tangential_pos_num - min_tangential_pos_num + 1)));      
	}
    }

  std::cout << "end of forward projection\n";
  auto finished_actual_fw = std::chrono::high_resolution_clock::now();
  std::cout << "fw_total=" << std::chrono::duration_cast<std::chrono::milliseconds>(finished_actual_fw - started_actual_fw).count() << std::endl;

}

END_NAMESPACE_STIR
