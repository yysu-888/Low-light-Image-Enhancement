#pragma once
#include <pybind11/pybind11.h>      // 导入 pybind11 的 C++ 库
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
namespace py = pybind11;

template<>
struct pybind11::detail::type_caster<cv::Point>{

    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));
    bool load(handle obj, bool){        
        if(!py::isinstance<py::tuple>(obj)){            
            std::logic_error("Point(x,y) should be a tuple!");            
            return false;       
        }       
 
        py::tuple pt = reinterpret_borrow<py::tuple>(obj);        
        if(pt.size()!=2){            
            std::logic_error("Point(x,y) tuple should be size of 2");            
            return false;        
        }       

        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());       
        return true;    
    }
  
    static handle cast(const cv::Point& pt, return_value_policy, handle){       
        return py::make_tuple(pt.x, pt.y).release();   
    }
};

template<>
struct pybind11::detail::type_caster<cv::Rect>{    

    PYBIND11_TYPE_CASTER(cv::Rect, _("tuple_xywh"));  
  
    bool load(handle obj, bool){        
        if(!py::isinstance<py::tuple>(obj)){            
            std::logic_error("Rect should be a tuple!");            
            return false;        
        }        

        py::tuple rect = reinterpret_borrow<py::tuple>(obj);       
        if(rect.size()!=4){            
            std::logic_error("Rect (x,y,w,h) tuple should be size of 4");            
            return false;        
        }      
  
        value = cv::Rect(rect[0].cast<int>(), rect[1].cast<int>(), rect[2].cast<int>(), rect[3].cast<int>());        
        return true;   
     }    

    static handle cast(const cv::Rect& rect, return_value_policy, handle){        
        return py::make_tuple(rect.x, rect.y, rect.width, rect.height).release();    
    }
};

template<>
struct pybind11::detail::type_caster<cv::Mat>{
public:   
 
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray")); 
   
    //! 1. cast numpy.ndarray to cv::Mat    
    bool load(handle obj, bool){        
        array b = reinterpret_borrow<array>(obj);        
        buffer_info info = b.request();    
    
        int nh = 1;        
        int nw = 1;        
        int nc = 1;        
        int ndims = info.ndim;        
        if(ndims == 2){           
           nh = info.shape[0];           
           nw = info.shape[1];       
        } 
        else if(ndims == 3){            
            nh = info.shape[0];           
            nw = info.shape[1];           
            nc = info.shape[2];        
        }else{            
            throw std::logic_error("Only support 2d, 2d matrix");            
            return false;       
        }       

        int dtype;        
        if(info.format == format_descriptor<unsigned char>::format()){            
            dtype = CV_8UC(nc);        
        }else if (info.format == format_descriptor<int>::format()){            
            dtype = CV_32SC(nc);       
        }else if (info.format == format_descriptor<float>::format()){           
            dtype = CV_32FC(nc);        
        }else{            
            throw std::logic_error("Unsupported type, only support uchar, int32, float"); 
            return false;
        }   
        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;    
    }    

    static handle cast(const cv::Mat& mat, return_value_policy, handle defval){        
        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type)? 2 : 3;
        if(depth == CV_8U){
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }else if(depth == CV_32S){
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }else if(depth == CV_32F){
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }else{            
            throw std::logic_error("Unsupport type, only support uchar, int32, float");
        }        

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) {
            bufferdim = {(size_t) nh, (size_t) nw};
            strides = {elemsize * (size_t) nw, elemsize};
        } else if (dim == 3) {
            bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
            strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
        }
        return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();    
}};