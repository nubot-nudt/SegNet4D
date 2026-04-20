// #include <pybind11/pybind11.h>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<cmath>
#include<omp.h>
#include<pybind11/numpy.h>

// #include <chrono>

#include <iostream>


namespace py = pybind11;

//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
py::array_t<int> find_features_by_bbox_with_yaw(py::array_t<int>& arr1, py::array_t<float>& arr2, py::array_t<int>& arr3) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    auto r3 = arr3.mutable_unchecked<2>();

    unsigned int arr1_length=arr1.shape()[0];
    unsigned int arr2_length=arr2.shape()[0];
    omp_set_num_threads(30);
    #pragma omp parallel for
    for (int i=0;i<arr2_length;i++)
    {
        float center[3];
        float extend[3];
        float theta;
        float centered[3];
        float rotated_point[3];
        int first_point[3];
        char first_flag;
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2);
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        theta = r2(i,6);
        float  cos_theta = cos(theta);
        float sin_theta = sin(theta);
        first_flag=0;
        // omp_set_num_threads(30);
        // #pragma omp parallel for
        for(int j=0;j<arr1_length;j++)
        {

            if((first_flag==1)&&(r1(j,0)>(first_point[0]+extend[0])||r1(j,0)<(first_point[0]-extend[0])||r1(j,1)>(first_point[1]+extend[1])||r1(j,1)<(first_point[1]-extend[1])||r1(j,2)>(first_point[2]+extend[2])||r1(j,2)<(first_point[2]-extend[2])))
            {
                    continue;
            }
            //去中心化，让这个bbox为中心
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
            //乘以旋转矩阵
            rotated_point[0] = centered[0]*cos_theta + centered[1]*sin_theta;
            rotated_point[1] = -centered[0]*sin_theta+ centered[1]*cos_theta;
            if((rotated_point[0]<=extend[0]/2)&&(rotated_point[0]>=-extend[0]/2) \
                &&(rotated_point[1]<=extend[1]/2)&&(rotated_point[1]>=-extend[1]/2) \
                    &&(centered[2]<=extend[2]/2)&&(centered[2]>=-extend[2]/2))
            {
                if(int(r2(i,7))>0)
                {
                    r3(j,int(r2(i,7))-1)=1;
                }
                if(first_flag==0)
                {
                    first_flag=1;
                    first_point[0] = r1(j,0);
                    first_point[1] = r1(j,1);
                    first_point[2] = r1(j,2);
                    
                }
            }
        }
    }
    return arr3;
}
//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
py::array_t<int> find_features_by_bbox_in_point_with_yaw(py::array_t<float>& arr1, py::array_t<float>& arr2, py::array_t<int>& arr3) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    auto r3 = arr3.mutable_unchecked<2>();

    unsigned int arr1_length=arr1.shape()[0];
    unsigned int arr2_length=arr2.shape()[0];
    omp_set_num_threads(30);
    #pragma omp parallel for
    for (int i=0;i<arr2_length;i++)
    {
        float center[3];
        float extend[3];
        float theta;
        float centered[3];
        float rotated_point[3];
        float first_point[3];
        char first_flag;
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2);
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        theta = r2(i,6);
        float  cos_theta = cos(theta);
        float sin_theta = sin(theta);
        first_flag=0;
        // omp_set_num_threads(30);
        // #pragma omp parallel for
        for(int j=0;j<arr1_length;j++)
        {

            if((first_flag==1)&&(r1(j,0)>(first_point[0]+extend[0])||r1(j,0)<(first_point[0]-extend[0])||r1(j,1)>(first_point[1]+extend[1])||r1(j,1)<(first_point[1]-extend[1])||r1(j,2)>(first_point[2]+extend[2])||r1(j,2)<(first_point[2]-extend[2])))
            {
                    continue;
            }
            //去中心化，让这个bbox为中心
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
            //乘以旋转矩阵
            rotated_point[0] = centered[0]*cos_theta + centered[1]*sin_theta;
            rotated_point[1] = -centered[0]*sin_theta+ centered[1]*cos_theta;
            if((rotated_point[0]<=extend[0]/2)&&(rotated_point[0]>=-extend[0]/2) \
                &&(rotated_point[1]<=extend[1]/2)&&(rotated_point[1]>=-extend[1]/2) \
                    &&(centered[2]<=extend[2]/2)&&(centered[2]>=-extend[2]/2))
            {
                if(int(r2(i,7))>0)
                {
                    r3(j,int(r2(i,7))-1)=1;
                }
                if(first_flag==0)
                {
                    first_flag=1;
                    first_point[0] = r1(j,0);
                    first_point[1] = r1(j,1);
                    first_point[2] = r1(j,2);
                    
                }
            }
        }
    }
    return arr3;
}

//*2d find poin in boundingboox and return idx 
//*arr1 voxel
//*arr2 bbox (N,7)
py::array_t<int> find_point_in_instance_bbox_with_yaw(py::array_t<float>& arr1, py::array_t<float>& arr2, py::array_t<int>& arr3,float out_ground) {
    auto r1 = arr1.unchecked<2>();
    auto r2 = arr2.unchecked<2>();
    auto r3 = arr3.mutable_unchecked<2>();

    unsigned int arr1_length=arr1.shape()[0];
    unsigned int arr2_length=arr2.shape()[0];
    omp_set_num_threads(30);
    #pragma omp parallel for
    for (int i=0;i<arr2_length;i++)
    {
        float center[3];
        float extend[3];
        float theta;
        float centered[3];
        float rotated_point[3];
        float first_point[3];
        char first_flag;
        //取出bbox的中心
        center[0] = r2(i,0);
        center[1] = r2(i,1);
        center[2] = r2(i,2)+out_ground;
        extend[0] = r2(i,3);
        extend[1] = r2(i,4);
        extend[2] = r2(i,5);
        theta = r2(i,6);
        float  cos_theta = cos(theta);
        float sin_theta = sin(theta);
        first_flag=0;
        // if(center[0]<5 &&center[0]>-5 && center[1]<5 && center[1]>-5)
        // {
        //     out_ground = 0;
        // }
        // omp_set_num_threads(30);
        // #pragma omp parallel for
        for(int j=0;j<arr1_length;j++)
        {
            // 这里是有问题的
            // if((first_flag==1)&&(r1(j,0)>(first_point[0]+extend[0])||r1(j,0)<(first_point[0]-extend[0])||r1(j,1)>(first_point[1]+extend[1])||r1(j,1)<(first_point[1]-extend[1])||r1(j,2)>(first_point[2]+extend[2])||r1(j,2)<(first_point[2]-extend[2])))
            // {
            //         continue;
            // }
            //去中心化，让这个bbox为中心
            centered[0] = r1(j,0) - center[0];
            centered[1] = r1(j,1) - center[1];
            centered[2] = r1(j,2) - center[2];
            //乘以旋转矩阵
            rotated_point[0] = centered[0]*cos_theta + centered[1]*sin_theta;
            rotated_point[1] = -centered[0]*sin_theta+ centered[1]*cos_theta;
            if((rotated_point[0]<=extend[0]/2)&&(rotated_point[0]>=-extend[0]/2) \
                &&(rotated_point[1]<=extend[1]/2)&&(rotated_point[1]>=-extend[1]/2) \
                    &&(centered[2]<=extend[2]/2)&&(centered[2]>=-extend[2]/2))
            {
                if(int(r2(i,7))>0)
                {
                    r3(j,int(r2(i,7))-1)=i+1;
                }
                if(first_flag==0)
                {
                    first_flag=1;
                    first_point[0] = r1(j,0);
                    first_point[1] = r1(j,1);
                    first_point[2] = r1(j,2);
                    
                }
            }
        }
    }
    return arr3;
}


py::array_t<float> pointcloud2bev(py::array_t<float> pointcloud,py::array_t<float> limit_range ,float grid_size_in, py::array_t<float> bev_init){
    //pointcloud: [N , 3]
    //limit_range: [x_min , y_min , z_min , x_max ,y_max , z_max]

    auto pc_r1 = pointcloud.unchecked<2>();
    auto lr = limit_range.unchecked<1>();
    auto bev_data = bev_init.mutable_unchecked<2>();

    // check pointcloud dim: Nx3
    if(pointcloud.ndim() !=2 || pointcloud.shape()[1] != 3){
        throw std::runtime_error("Input pointcloud should have shape (N, 3).");
    }

    // get pointcloud ptr
    float x_min = lr(0); float y_min = lr(1); float z_min = lr(2);
    float x_max = lr(3); float y_max = lr(4); float z_max = lr(5);

    float grid_size = grid_size_in;

    // cal bev height and width
    int height  = int((x_max - x_min)/grid_size);
    int width   = int((y_max - y_min)/grid_size);

    // projection to bev space
    int arr1_length=pointcloud.shape()[0];

    omp_set_num_threads(10);
    #pragma omp parallel for
    for(int i=0;i <arr1_length;i++){
        float x = pc_r1(i,0);
        float y = pc_r1(i,1);
        float z = pc_r1(i,2);

        // check point edge
        if(x > x_max || x < x_min || y > y_max || y < y_min) continue;

        int x_index = int ((-x - x_min)/grid_size);
        int y_index = int ((-y - y_min)/grid_size);


        if(z > bev_data(x_index, y_index))
            bev_data(x_index, y_index) = z;
            
    }
    return bev_init;
}


py::array_t<float> pointcloud2bevHeight(py::array_t<float> pointcloud, 
                                        py::array_t<float> limit_range, 
                                        py::array_t<float> bev_init, 
                                        py::array_t<float> bev_init_final, 
                                        py::array_t<float> bev_init_count,
                                        py::array_t<float> bev_max, 
                                        py::array_t<float> bev_min, 
                                        float grid_size_in,
                                        int max_num_in_voxel)
{
    // pointcloud: [N , 3]
    // limit_range: [x_min , y_min , z_min , x_max ,y_max , z_max]

    auto pc_r1 = pointcloud.unchecked<2>();
    auto lr = limit_range.unchecked<1>();
    auto bev_data = bev_init.mutable_unchecked<2>();
    auto bev_data_final = bev_init_final.mutable_unchecked<2>();
    auto bev_data_count = bev_init_count.mutable_unchecked<2>();

    // 多线程中使用它要快一点
    auto bev_max_height = bev_max.mutable_unchecked<2>();
    auto bev_min_height = bev_min.mutable_unchecked<2>();

    // check pointcloud dim: Nx3
    if (pointcloud.ndim() != 2 || pointcloud.shape()[1] != 3)
    {
        throw std::runtime_error("Input pointcloud should have shape (N, 3).");
    }

    // get pointcloud ptr
    float x_min = lr(0);
    float y_min = lr(1);
    float z_min = lr(2);
    float x_max = lr(3);
    float y_max = lr(4);
    float z_max = lr(5);

    float grid_size = grid_size_in;

    // cal bev height and width
    int height = int((x_max - x_min) / grid_size);
    int width = int((y_max - y_min) / grid_size);

    // projection to bev space
    int arr1_length = pointcloud.shape()[0];

    // std::vector<float> bev_max_height(height * width, z_min);
    // std::vector<float> bev_min_height(height * width, z_max);
    // std::vector<std::vector<float>> bev_voxel(width*height);
    omp_set_num_threads(10);
    #pragma omp parallel for
    for (int i = 0; i < arr1_length; i++)
    {
        float x = pc_r1(i, 0);
        float y = pc_r1(i, 1);
        float z = pc_r1(i, 2);

        // check point edge
        if (x > x_max || x < x_min || y > y_max || y < y_min || z > z_max || z < z_min)
            continue;

        int x_index = int((-x - x_min) / grid_size);
        int y_index = int((-y - y_min) / grid_size);

        // if(z > bev_data(x_index, y_index))
        //     bev_data(x_index, y_index) = z;
        
        if (z > bev_max_height(x_index,y_index))
        {
            bev_max_height(x_index,y_index) = z;
            bev_data(x_index, y_index) = z - bev_min_height(x_index,y_index);
            bev_data_count(x_index,y_index) ++;
            if(bev_data_count(x_index,y_index)>max_num_in_voxel) 
                bev_data_final(x_index,y_index) = bev_data(x_index, y_index);
        }
        if (z < bev_min_height(x_index,y_index))
        {
            bev_min_height(x_index,y_index) = z;
            bev_data(x_index, y_index) = bev_max_height(x_index,y_index) - z;
            bev_data_count(x_index,y_index) ++;
            if(bev_data_count(x_index,y_index)>max_num_in_voxel) 
                bev_data_final(x_index,y_index) = bev_data(x_index, y_index);         
        }
    }
    return bev_init_final;
}


py::array_t<float> pointcloud2bevHeightSingleThread_2(py::array_t<float> pointcloud, 
                                                    py::array_t<float> limit_range, 
                                                    py::array_t<float> bev_init, 
                                                    py::array_t<float> bev_init_final, 
                                                    py::array_t<float> bev_init_count,
                                                    py::array_t<float> bev_max, 
                                                    py::array_t<float> bev_min, 
                                                    float grid_size_in,
                                                    int max_num_in_voxel_in)
{
    // pointcloud: [N , 3]
    // limit_range: [x_min , y_min , z_min , x_max ,y_max , z_max]

    auto pc_r1 = pointcloud.unchecked<2>();
    auto lr = limit_range.unchecked<1>();
    auto bev_data = bev_init.mutable_unchecked<2>();
    auto bev_data_final = bev_init_final.mutable_unchecked<2>();
    auto bev_max_height = bev_max.mutable_unchecked<2>();
    auto bev_min_height = bev_min.mutable_unchecked<2>();
    auto bev_data_count = bev_init_count.mutable_unchecked<2>();

    // check pointcloud dim: Nx3
    if (pointcloud.ndim() != 2 || pointcloud.shape()[1] != 3)
    {
        throw std::runtime_error("Input pointcloud should have shape (N, 3).");
    }

    // get pointcloud ptr
    float x_min = lr(0);
    float y_min = lr(1);
    float z_min = lr(2);
    float x_max = lr(3);
    float y_max = lr(4);
    float z_max = lr(5);

    float grid_size = grid_size_in;
    int max_num_in_voxel = max_num_in_voxel_in;

    // cal bev height and width
    int height = int((x_max - x_min) / grid_size);
    int width = int((y_max - y_min) / grid_size);

    // projection to bev space
    int arr1_length = pointcloud.shape()[0];

    // std::vector<float> bev_max_height(height * width, z_min);
    // std::vector<float> bev_min_height(height * width, z_max);
    // std::vector<int> bev_count(height * width,0);
    for (int i = 0; i < arr1_length; i++)
    {
        float x = pc_r1(i, 0);
        float y = pc_r1(i, 1);
        float z = pc_r1(i, 2);

        // check point edge
        if (x > x_max || x < x_min || y > y_max || y < y_min || z > z_max || z < z_min)
            continue;

        int x_index = int((-x - x_min) / grid_size);
        int y_index = int((-y - y_min) / grid_size);

        // if(z > bev_data(x_index, y_index))
        //     bev_data(x_index, y_index) = z;
        // if (z > bev_max_height[x_index* width+y_index])
        // {
        //     bev_max_height[x_index* width+y_index] = z;
        //     bev_data(x_index, y_index) = z - bev_min_height[x_index* width+ y_index];
        //     bev_count[x_index* width+y_index]++;
        //     if(bev_count[x_index* width+y_index]>num_max_in_voxel)
        //         bev_data_final(x_index, y_index) = bev_data(x_index, y_index);
        // }
        // if (z < bev_min_height[x_index* width+y_index])
        // {
        //     bev_min_height[x_index* width+y_index] = z;
        //     bev_data(x_index, y_index) = bev_max_height[x_index* width+y_index] - z;
        //     bev_count[x_index* width+y_index]++;
        //     if(bev_count[x_index* width+y_index]>num_max_in_voxel)
        //         bev_data_final(x_index, y_index) = bev_data(x_index, y_index);
        // }
        if (z > bev_max_height(x_index,y_index))
        {
            bev_max_height(x_index,y_index) = z;
            bev_data(x_index, y_index) = z - bev_min_height(x_index,y_index);
            bev_data_count(x_index,y_index) = bev_data_count(x_index,y_index) + 1;
            if(bev_data_count(x_index,y_index)>max_num_in_voxel) {
                // std::cout<<"bev_data_count:"<<bev_data_count(x_index,y_index)<<std::endl;
                // std::cout<<"max_num_in_voxel:"<<max_num_in_voxel<<std::endl;
                bev_data_final(x_index,y_index) = bev_data(x_index, y_index);
            }
        }
        if (z < bev_min_height(x_index,y_index))
        {
            bev_min_height(x_index,y_index) = z;
            bev_data(x_index, y_index) = bev_max_height(x_index,y_index) - z;
            bev_data_count(x_index,y_index) = bev_data_count(x_index,y_index) + 1;
            if(bev_data_count(x_index,y_index)>max_num_in_voxel) {
                // std::cout<<"bev_data_count:"<<bev_data_count(x_index,y_index)<<std::endl;
                // std::cout<<"max_num_in_voxel:"<<max_num_in_voxel<<std::endl;
                bev_data_final(x_index,y_index) = bev_data(x_index, y_index); 
            }     
        }
    }
    return bev_init_final;
}


py::array_t<float> pointcloud2bevHeightSingleThread_1(py::array_t<float> pointcloud, py::array_t<float> limit_range, py::array_t<float> bev_init, py::array_t<float> bev_max, py::array_t<float> bev_min, float grid_size_in)
{
    // pointcloud: [N , 3]
    // limit_range: [x_min , y_min , z_min , x_max ,y_max , z_max]

    auto pc_r1 = pointcloud.unchecked<2>();
    auto lr = limit_range.unchecked<1>();
    auto bev_data = bev_init.mutable_unchecked<2>();
    auto bev_max_height = bev_max.mutable_unchecked<2>();
    auto bev_min_height = bev_min.mutable_unchecked<2>();

    // check pointcloud dim: Nx3
    if (pointcloud.ndim() != 2 || pointcloud.shape()[1] != 3)
    {
        throw std::runtime_error("Input pointcloud should have shape (N, 3).");
    }

    // get pointcloud ptr
    float x_min = lr(0);
    float y_min = lr(1);
    float z_min = lr(2);
    float x_max = lr(3);
    float y_max = lr(4);
    float z_max = lr(5);

    float grid_size = grid_size_in;

    // cal bev height and width
    int height = int((x_max - x_min) / grid_size);
    int width = int((y_max - y_min) / grid_size);

    // projection to bev space
    int arr1_length = pointcloud.shape()[0];

    // std::vector<float> bev_max_height(height * width, z_min);
    // std::vector<float> bev_min_height(height * width, z_max);
    for (int i = 0; i < arr1_length; i++)
    {
        float x = pc_r1(i, 0);
        float y = pc_r1(i, 1);
        float z = pc_r1(i, 2);

        // check point edge
        if (x > x_max || x < x_min || y > y_max || y < y_min || z > z_max || z < z_min)
            continue;

        int x_index = int((-x - x_min) / grid_size);
        int y_index = int((-y - y_min) / grid_size);

        // if(z > bev_data(x_index, y_index))
        //     bev_data(x_index, y_index) = z;
        if (z > bev_max_height(x_index, y_index))
        {
            bev_max_height(x_index, y_index) = z;
            bev_data(x_index, y_index) = z - bev_min_height(x_index, y_index);
        }
        if (z < bev_min_height(x_index, y_index))
        {
            bev_min_height(x_index, y_index) = z;
            bev_data(x_index, y_index) = bev_max_height(x_index, y_index) - z;
        }
    }
    return bev_init;
}


py::array_t<float> bev2index(py::array_t<float> bev, py::array_t<float> bev_index)
{
    // pointcloud: [N , 3]
    // limit_range: [x_min , y_min , z_min , x_max ,y_max , z_max]

    auto bev_data = bev.mutable_unchecked<2>();
    auto bev_return = bev_index.mutable_unchecked<2>();


    unsigned int bev_height=bev.shape()[0];
    unsigned int bev_width=bev.shape()[1];

    for (int i = 0; i < bev_height; i++)
    {
        for(int j=0; j< bev_width;j++)
        {
            bev_return(i*bev_width+j,0) = i;
            bev_return(i*bev_width+j,1) = j;
            bev_return(i*bev_width+j,2) = bev_data(i,j);
        }
    }
    return bev_index;
}

PYBIND11_MODULE(Array_Index, m)
{
    // 可选，说明这个模块是做什么的
    m.doc() = "pybind11 example plugin";
    //def( "给python调用方法名"， &实际操作的函数， "函数功能说明" ). 其中函数功能说明为可选
    m.def("find_features_by_bbox_with_yaw", &find_features_by_bbox_with_yaw, "A function return array idx other array "); //
    m.def("find_features_by_bbox_in_point_with_yaw", &find_features_by_bbox_in_point_with_yaw, "A function return array idx other array ");
    m.def("find_point_in_instance_bbox_with_yaw", &find_point_in_instance_bbox_with_yaw, "A function return array idx other array "); //

    // convert pointcloud to bev
    m.def("pointcloud2bev", &pointcloud2bev, "Convert point cloud to BEV image");
    m.def("bev2index", &bev2index, "Get BEV index");
    m.def("pointcloud2bevHeight", &pointcloud2bevHeight, "Convert point cloud to BEV image");
    m.def("pointcloud2bevHeightSingleThread_1", &pointcloud2bevHeightSingleThread_1, "Convert point cloud to BEV image");
    m.def("pointcloud2bevHeightSingleThread_2", &pointcloud2bevHeightSingleThread_2, "Convert point cloud to BEV image");
}




