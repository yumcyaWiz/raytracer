#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <memory>
#include <string>

#include <random>
#include <ctime>

#include <chrono>

#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


//Width, Height, Sample
const int W = 800;
const int H = 800;
const int N = 100;


//Vector Class
class Vec3 {
    public:
        double x;
        double y;
        double z;
        Vec3() {x = 0; y = 0; z = 0;};
        Vec3(double _x) {x = _x; y = _x; z = _x;};
        Vec3(double _x, double _y, double _z) {x = _x; y = _y; z = _z;};
        ~Vec3() {};
        
        Vec3 operator-() {return Vec3(-x, -y, -z);};
        void operator+=(Vec3 v) {x += v.x; y += v.y; z += v.z;};
        double dot(Vec3 v) {return x*v.x + y*v.y + z*v.z;};
        Vec3 cross(Vec3 v) {return Vec3(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);};
       
        double length() {return std::sqrt(x*x + y*y + z*z);};
        double length2() {return x*x + y*y + z*z;};
        double norm1() { return std::abs(x) + std::abs(y) + std::abs(z); };
        double normInf() { return std::max(std::abs(x), std::max(std::abs(y), std::abs(z))); };
        Vec3 normalize();
    
        void print() {
            std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
        }
};
Vec3 operator+ (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
};
Vec3 operator- (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
};
Vec3 operator* (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
};
Vec3 operator* (const Vec3& v1, double k) {
    return Vec3(v1.x * k, v1.y * k, v1.z * k);
};
Vec3 operator* (double k, const Vec3& v2) {
    return Vec3(k * v2.x, k * v2.y, k * v2.z);
};
Vec3 operator/ (const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
Vec3 operator/ (const Vec3& v1, double k) {
    return Vec3(v1.x / k, v1.y / k, v1.z / k);
};
Vec3 operator/ (double k, const Vec3& v2) {
    return Vec3(v2.x / k, v2.y / k, v2.z / k);
};
Vec3 Vec3::normalize() {return Vec3(x, y, z)/std::sqrt(x*x + y*y + z*z);};


inline Vec3 reflect(Vec3 v, Vec3 n) {
    return v + 2.0*(-v).dot(n)*n;
}
inline bool refract(Vec3 v, Vec3 n, double n1, double n2, Vec3& res) {
    double eta = n1/n2;
    double cosI = (-v).dot(n);
    double sin2I = std::max(0.0, 1.0 - cosI*cosI);
    double sin2T = eta*eta*sin2I;
    if(sin2T >= 1.0)
        return false;
    double cosT = std::sqrt(1.0 - sin2T);
    res = eta*v + (eta*cosI - cosT)*n;
    return true;
}


//RGB Class used for storing a color value
inline double clamp(double v, double min, double max) {
    if(v > max) {
        return max;
    }
    else if(v < min) {
        return min;
    }
    else {
        return v;
    }
}
class RGB {
    public:
        double r;
        double g;
        double b;
        RGB() {r = 0.0; g = 0.0; b = 0.0;};
        RGB(double _r) {r = _r; g = _r; b = _r;};
        RGB(double _r, double _g, double _b) {r = _r; g = _g; b = _b;};
        RGB(const Vec3& v) {r = v.x; g = v.y; b = v.z;};
        ~RGB() {};
        void rgb_clamp() { r = clamp(r, 0.0, 255.0); g = clamp(g, 0.0, 255.0); b = clamp(b, 0.0, 255.0); };
};
RGB operator+ (const RGB& r1, const RGB& r2) {
    return RGB(r1.r + r2.r, r1.g + r2.g, r1.b + r2.b);
}
RGB operator- (const RGB& r1, const RGB& r2) {
    return RGB(r1.r - r2.r, r1.g - r2.g, r1.b - r2.b);
}
RGB operator* (const RGB& r1, const RGB& r2) {
    return RGB(r1.r * r2.r, r1.g * r2.g, r1.b * r2.b);
}
RGB operator* (const RGB& r1, double k) {
    return RGB(r1.r * k, r1.g * k, r1.b * k);
}
RGB operator* (double k, const RGB& r2) {
    return RGB(k * r2.r, k * r2.g, k * r2.b);
}
RGB operator/ (const RGB& r1, const RGB& r2) {
    return RGB(r1.r / r2.r, r1.g / r2.g, r1.b / r2.b);
}
RGB operator/ (const RGB& r1, double k) {
    return RGB(r1.r / k, r1.g / k, r1.b / k);
}
RGB operator/ (double k, const RGB& r2) {
    return RGB(r2.r / k, r2.g / k, r2.b / k);
}


//Ray Class o:origin, d:direction
class Ray {
    public:
        Vec3 o;
        Vec3 d;
        Ray() {};
        Ray(const Vec3& _o, const Vec3& _d) {o = _o; d = _d;};
        double tmin = 0.00001;
        double tmax = 10000.0;

        void print() {
            std::cout << "origin: ";
            o.print();
            std::cout << ", direction: ";
            d.print();
            std::cout << std::endl;
        };
};


//Material Class
//type: material type
class Material {
    public:
        std::string type;
        RGB reflectance;
        double IOR;
        RGB emitColor;
};
class Diffuse : public Material {
    public:
        Diffuse(const RGB& _r) {
            type = "diffuse";
            reflectance = _r;
        };
};
class Metalistic : public Material {
    public:
        Metalistic(const RGB& _r) {
            type = "metalistic";
            reflectance = _r;
        };
};
class Emissive : public Material {
    public:
        Emissive(const RGB& _emitColor) { type = "emissive"; emitColor = _emitColor; };
};
class Mirror : public Material {
    public:
        Mirror(const RGB& _r) {
            type = "mirror";
            reflectance = _r;
        };
};
class Glass : public Material {
    public:
        Glass(const RGB& _r, double _IOR) {
            type = "glass";
            reflectance = _r;
            IOR = _IOR;
        };
};
inline double fresnel(double n1, double n2, Vec3 v, Vec3 n) {
    double f0 = std::pow((n1 - n2)/(n1 + n2), 2.0);
    return f0 + (1.0 - f0)*std::pow(1.0 - (-v).dot(n), 5.0);
}


//Hit Class holds various information at the intersection point
//hitPos: hit position
//hitNormal: normal vector at the hit position
//hitMaterial: material at the hit position
class Hit {
    public:
        double t;
        bool inside;
        Vec3 rayDir;
        Vec3 hitPos;
        Vec3 hitNormal;
        Material hitMaterial;
        Hit() {};
        Hit(double _t, bool _inside, const Vec3& _rayDir, const Vec3& _hitPos, const Vec3& _hitNormal, const Material& _hitMaterial) {
            t = _t;
            inside = _inside;
            rayDir = _rayDir;
            hitPos = _hitPos;
            hitNormal = _hitNormal;
            hitMaterial = _hitMaterial;
        };
        ~Hit() {};
};


//Object Class represents geometry information and provides ray-object intersection function
class Object {
    public:
        Material mat;
        virtual bool intersect(Ray &r, Hit& res) = 0;
};


class Sphere : public Object {
    public:
        Vec3 center;
        double radius;
        Sphere(const Vec3& _center, double _radius, const Material& _mat) {
            center = _center;
            radius = _radius;
            mat = _mat;
        };
        bool intersect(Ray& r, Hit& res) {
            double b = (r.o - center).dot(r.d);
            double c = (r.o - center).length2() - radius*radius;
            double D = b*b - c;
            if(D >= 0) {
                double t1 = std::min(-b + sqrt(D), -b - sqrt(D));
                double t2 = std::max(-b + sqrt(D), -b - sqrt(D));
                //std::cout << t1 << ", " << t2 << std::endl;
                if(t1 < r.tmin) {
                    if(t2 > r.tmin && t2 < r.tmax)
                        res.t = t2;
                    else
                        return false;
                }
                else {
                    if(t1 < r.tmax)
                        res.t = t1;
                    else if(t2 > r.tmin && t2 < r.tmax)
                        res.t = t2;
                    else
                        return false;
                }
                //std::cout << res.t << std::endl;

                res.rayDir = r.d;
                res.hitPos = r.o + r.d*res.t;
                res.hitNormal = (res.hitPos - center).normalize();
                res.hitMaterial = mat;
                res.inside = (-r.d).dot(res.hitNormal) < 0;
                return true;
            }
            else {
                return false;
            }
        };
};
class Plane : public Object {
    public:
        Vec3 center;
        Vec3 normal;
        double size;
        Plane(const Vec3& _center, const Vec3& _normal, double _size, const Material& _mat) {
            center = _center;
            normal = _normal;
            size = _size;
            mat = _mat;
        };
        bool intersect(Ray& ray, Hit& res) {
            if(ray.d.dot(normal) == 0) {
                return false;
            }
            res.t = (center - ray.o).dot(normal)/(ray.d.dot(normal));
            if(res.t < ray.tmin || res.t > ray.tmax) {
                return false;
            }
            res.hitPos = ray.o + ray.d*res.t;
            if((res.hitPos - center).normInf() > size) {
                return false;
            }
            res.hitNormal = normal;
            res.hitMaterial = mat;
            res.inside = false;
            return true;
        };
};
class Box : public Object {
    public:
        Vec3 center;
        double radius;
        Box(const Vec3& _center, double _radius, const Material& _mat) {
            center = _center;
            radius = _radius;
            mat = _mat;
        };
        bool intersect(const Ray& r, Hit& res) {
            return false;
        }
};


//Scene Class holds all the geometry information in the scene
//ray-scene intersection function provided
class Scene {
    public:
        std::vector<std::unique_ptr<Object>> objects;
        Scene() {};
        void add(Object* o) {objects.push_back(std::unique_ptr<Object>(o));};
        bool trace(Ray& r, Hit& res) {
            bool hitFrag = false;
            for(int i = 0; i < objects.size(); i++) {
                Object *obj = objects[i].get();
                Hit res2;
                if(obj->intersect(r, res2)) {
                    if((res2.t < res.t && res2.t >= r.tmin) || !hitFrag) res = res2;
                    hitFrag = true;
                }
            }
            return hitFrag;
        };
};


//Image Class holds RGB color at each pixel
//ppm output functionality provided
inline RGB pow(RGB r, double k) {
    return RGB(pow(r.r, k), pow(r.g, k), pow(r.b, k));
}
inline int clamp(int x, int min, int max) {
    if(x < min) 
        return min;
    else if(x > max)
        return max;
    else
        return x;
}
class Image {
    public:
        RGB* color;
        int width;
        int height;
        Image(int _width, int _height) {
            width = _width;
            height = _height;
            color = new RGB[width*height];
        };
        ~Image() {
            delete[] color;
        };
        void setPixel(int i, int j, RGB col) { color[i + j*width] = col; };
        RGB getPixel(int i, int j) { return color[i + j*width]; };
        void ppm_output(std::string filename) {
            std::ofstream file(filename);

            file << "P3\n";
            file << width << " " << height << "\n";
            file << 255 << "\n";

            for(int j = 0; j < height; j++) {
                for(int i = 0; i < width; i++) {
                    RGB col = color[i + j*width];
                    col.rgb_clamp();
                    int r = clamp((int)(255*col.r), 0, 255);
                    int g = clamp((int)(255*col.g), 0, 255);
                    int b = clamp((int)(255*col.b), 0, 255);
                    file << r << " " << g << " " << b << "\n";
                }
            }
            file.close();
        };
        void divide(double k) {
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                    color[i + j*width] = color[i + j*width]/k;
        };
        void gamma_correlation() {
            for(int i = 0; i < width; i++)
                for(int j = 0; j < height; j++)
                    color[i + j*width] = pow(color[i + j*width], 1.0/2.2);
        };
};


//noise function
std::random_device rnd;
std::mt19937 mtrnd(rnd());
std::uniform_real_distribution<double> dist(0, 1);
inline double noise() {
    return dist(mtrnd);
}


//random vector in the unit sphere
Vec3 random_in_unitSphere() {
    Vec3 v;
    do {
        v = Vec3(2.0*noise() - 1.0, 2.0*noise() - 1.0, 2.0*noise() - 1.0);
    }
    while(v.length2() >= 1.0);
    return v;
}


//sky color
RGB sky(const Ray& ray) {
    float t = 0.5 * (ray.d.y + 1.0);
    Vec3 col = (1.0 - t)*Vec3(1) + t*Vec3(0.5, 0.7, 1.0);
    return RGB(col);
}


inline double toDeg(double rad);
int HDRI_W;
int HDRI_H;
int HDRI_C;
float* HDRI = stbi_loadf("PaperMill_E_3k.hdr", &HDRI_W, &HDRI_H, &HDRI_C, 0);
RGB IBL(const Ray& ray) {
    double theta1 = std::atan2(ray.d.z, ray.d.x) + M_PI;
    double theta2 = std::atan(ray.d.y/(std::sqrt(ray.d.x*ray.d.x + ray.d.z*ray.d.z))) + M_PI/2.0;
    int offset = 1700;
    int u = ((int)(theta1/(2.0*M_PI)*HDRI_W) + offset) % HDRI_W;
    int v = HDRI_H - (int)(theta2/(M_PI)*HDRI_H);
    int adr = HDRI_C*u + HDRI_C*HDRI_W*v;
    return RGB(HDRI[adr], HDRI[adr+1], HDRI[adr+2]);
}
RGB checkerboard(double u, double v, double interval) {
    u = std::floor(2.0/interval*u);
    v = std::floor(2.0/interval*v);
    double p = std::fmod(u + v, 2.0);
    return p*RGB(0.1) + (1 - p)*RGB(0.9);
}
RGB IBL_test(const Ray& ray) {
    double theta1 = std::atan2(ray.d.z, ray.d.x) + M_PI;
    double theta2 = std::atan(ray.d.y/(std::sqrt(ray.d.x*ray.d.x + ray.d.z*ray.d.z))) + M_PI/2.0;
    return checkerboard(theta1, theta2, 0.2);
}


//shading function
double russian_roulette_prob;
RGB shading(Ray& ray, Scene& scene, int depth) {
    if(depth < 3) {
        russian_roulette_prob = 1.0;
    }
    else {
        russian_roulette_prob /= 1.2;
    }

    if(noise() > russian_roulette_prob) {
        return RGB(0);
    }

    Hit res = Hit();
    if(scene.trace(ray, res)) {
        Material mat = res.hitMaterial;
        if(mat.type == "diffuse") {
            Vec3 nextDir = (res.hitNormal + random_in_unitSphere()).normalize();
            Ray nextRay = Ray(res.hitPos, nextDir);
            return mat.reflectance * shading(nextRay, scene, depth + 1)/russian_roulette_prob;
        }
        else if(mat.type == "metalistic") {
            Vec3 nextDir = (reflect(ray.d, res.hitNormal) + 0.5*random_in_unitSphere()).normalize();
            Ray nextRay = Ray(res.hitPos, nextDir);
            return mat.reflectance * shading(nextRay, scene, depth + 1)/russian_roulette_prob;
        }
        else if(mat.type == "emissive") {
            return mat.emitColor;
        }
        else if(mat.type == "mirror") {
            Vec3 nextDir = reflect(ray.d, res.hitNormal);
            Ray nextRay = Ray(res.hitPos, nextDir);
            return mat.reflectance * shading(nextRay, scene, depth + 1)/russian_roulette_prob;
        }
        else if(mat.type == "glass") {
            Vec3 nextDir;
            Ray nextRay;
            double p;
            if(!res.inside) {
                p = fresnel(1.0, mat.IOR, ray.d, res.hitNormal);
                if(noise() < p) {
                    nextDir = reflect(ray.d, res.hitNormal);
                    nextRay = Ray(res.hitPos, nextDir);
                }
                else {
                    if(refract(ray.d, res.hitNormal, 1.0, mat.IOR, nextDir)) {
                        nextRay = Ray(res.hitPos, nextDir);
                    }
                    else {
                        nextDir = reflect(ray.d, res.hitNormal);
                        nextRay = Ray(res.hitPos, nextDir);
                    }
                }
            }
            else {
                p = fresnel(mat.IOR, 1.0, ray.d, -res.hitNormal); 
                if(noise() < p) {
                    nextDir = reflect(ray.d, -res.hitNormal);
                    nextRay = Ray(res.hitPos, nextDir);
                }
                else {
                    if(refract(ray.d, -res.hitNormal, mat.IOR, 1.0, nextDir)) {
                        nextRay = Ray(res.hitPos, nextDir);
                    }
                    else {
                        nextDir = reflect(ray.d, -res.hitNormal);
                        nextRay = Ray(res.hitPos, nextDir);
                    }
                }
            }
            return mat.reflectance * shading(nextRay, scene, depth + 1)/russian_roulette_prob;
        }
        else {
            return RGB(0);
        }
    }
    else {
        return IBL(ray);
    }
}


RGB shadingNEE(Ray& ray, Scene& scene, int depth) {
    if(depth < 3) {
        russian_roulette_prob = 1.0;
    }
    else {
        russian_roulette_prob /= 1.2;
    }

    if(noise() > russian_roulette_prob) {
        return RGB(0);
    }

    Hit res;
    if(scene.trace(ray, res)) {
        Material mat = res.hitMaterial;
        if(mat.type == "diffuse") {
        }
        else if(mat.type == "metalistic") {
        }
        else if(mat.type == "emissive") {
        }
        else if(mat.type == "mirror") {
        }
        else if(mat.type == "glass") {
        }
    }

    return RGB(0);
}


class Timer {
    public:
        std::chrono::system_clock::time_point t_start;
        std::chrono::system_clock::time_point t_end;
        Timer() {};
        ~Timer() {};
        void start() { t_start = std::chrono::system_clock::now(); };
        void stop() {
            t_end = std::chrono::system_clock::now();
            auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
            std::cout << msec << "ms\n";
        };
};


inline double toRad(double theta) {
    return 3.14/180.0 * theta;
}
inline double toDeg(double rad) {
    return 180.0/3.14 * rad;
}


struct Camera_ray_params {
    Vec3 sensorPos;
    Vec3 aperturePos;
    Camera_ray_params() {};
};
Vec3 random_in_unitDisk(const Vec3& right, const Vec3& up) {
    Vec3 v;
    do {
        v = (2.0*noise() - 1.0)*right + (2.0*noise() - 1.0)*up;
    }
    while(v.length2() > 1.0);
    return v;
}
class Camera {
    public:
        Vec3 sensor_center_pos;
        Vec3 sensor_front;
        Vec3 sensor_right;
        Vec3 sensor_up;
        Vec3 sensorPos;
        Vec3 aperturePos;
        Vec3 sensor_to_aperture;
        double aperture_distance;
        double object_distance;
        double focal_length;
        double aperture_radius;
        double sensor_size = 2;
        double f;
        Camera() {};
        Camera(const Vec3& _sensor_center_pos, const Vec3& _sensor_front, double _aperture_distance, double _focal_length, double _f) {
            sensor_center_pos = _sensor_center_pos;
            sensor_front = _sensor_front;
            sensor_right = sensor_front.cross(Vec3(0, 1, 0));
            sensor_up = -sensor_right.cross(sensor_front);

            sensor_front = sensor_front.normalize();
            sensor_right = sensor_right.normalize();
            sensor_up = sensor_up.normalize();
            
            aperture_distance = _aperture_distance;
            focal_length = _focal_length;
            f = _f;
            aperture_radius = focal_length/f;
            object_distance = 1.0/(1.0/focal_length - 1.0/aperture_distance);
        };
        Ray getRay(double u, double v, Camera_ray_params& params) {
            Vec3 sensorPos = sensor_center_pos + u*sensor_right + v*sensor_up;
            Vec3 aperturePos = sensor_center_pos + sensor_front*aperture_distance + aperture_radius*random_in_unitDisk(sensor_right, sensor_up);
            Vec3 sensor_to_aperture = (aperturePos - sensorPos).normalize();
            params.sensorPos = sensorPos;
            params.aperturePos = aperturePos;

            Vec3 sensor_to_aperture_center = ((sensor_center_pos + aperture_distance*sensor_front) - sensorPos).normalize();
            Vec3 objectPos = sensorPos + (aperture_distance + object_distance)/sensor_to_aperture_center.dot(sensor_front) * sensor_to_aperture_center;
            Vec3 rayDir = (objectPos - aperturePos).normalize();
            return Ray(aperturePos, rayDir);
        };
        void print() const {
            std::cout << "Camera Parameters" << std::endl;
            std::cout << "Aperture Distance : " << aperture_distance << std::endl;
            std::cout << "Object Plane Distance : " << object_distance << std::endl;
            std::cout << "Focal Length : " << focal_length << std::endl;
            std::cout << "Aperture Radius : " << aperture_radius << std::endl;
        }
};


std::string percentage(double x, double max) {
    return std::to_string(x/max*100) + "%";
}
std::string progressbar(double x, double max) {
    int max_count = 40;
    int cur_count = (int)(x/max*max_count);
    std::string str;
    str += "[";
    for(int i = 0; i < cur_count; i++) {
        str += "#";
    }
    for(int i = 0; i < (max_count - cur_count - 1); i++) {
        str += " ";
    }
    str += "]";
    return str;
}


void render_normal(Camera& cam, Scene& scene) {
    Image img(W, H);
    
    std::cout << "Normal Rendering" << std::endl;
    Timer t;
    t.start();
    for(int i = 0; i < W; i++) {
        for(int j = 0; j < H; j++) {
            double u = (2.0*((W - 1) - i) - W)/(double)H;
            double v = (2.0*((H - 1) - j) - H)/(double)H;
            Camera_ray_params params;
            Ray ray = cam.getRay(u, v, params);

            Hit res;
            if(scene.trace(ray, res)) {
                img.setPixel(i, j, RGB((res.hitNormal + 1.0)/2.0));
            }
            else {
                img.setPixel(i, j, RGB(0));
            }
        }
    }
    t.stop();

    img.ppm_output("output_normal.ppm");
}
void render_depth(Camera& cam, Scene& scene) {
    Image img(W, H);

    std::cout << "Depth Rendering" << std::endl;
    Timer t;
    t.start();
    for(int i = 0; i < W; i++) {
        for(int j = 0; j < H; j++) {
            double u = (2.0*((W - 1) - i) - W)/(double)H;
            double v = (2.0*((H - 1) - j) - H)/(double)H;
            Camera_ray_params params;
            Ray ray = cam.getRay(u, v, params);

            Hit res;
            if(scene.trace(ray, res))
                img.setPixel(i, j, RGB(1.0 - res.t/10.0));
            else
                img.setPixel(i, j, RGB(0));
        }
    }
    t.stop();

    img.ppm_output("output_depth.ppm");
}
void render(Camera cam, Scene& scene) {
    Image img(W, H);

    std::cout << "Rendering" << std::endl;
    std::cout << "width:" << W << " height:" << H << " sample:" << N << std::endl;
    Timer t;
    t.start();
    #pragma omp parallel for schedule(dynamic, 1)
    for(int k = 0; k < N; k++) {
        for(int i = 0; i < W; i++) {
            for(int j = 0; j < H; j++) {
                double u = (2.0*((W - 1) - i + noise()) - W)/(double)H;
                double v = (2.0*((H - 1) - j + noise()) - H)/(double)H;
                Camera_ray_params params;
                Ray ray = cam.getRay(u, v, params);
                RGB radiance = shading(ray, scene, 0);
                Vec3 sensor_to_aperture = (params.aperturePos - params.sensorPos);
                double cos_term = std::pow(cam.sensor_front.dot(sensor_to_aperture.normalize()), 2.0);
                RGB irradiance = radiance * cos_term / std::pow(sensor_to_aperture.length(), 2.0);
                img.setPixel(i, j, irradiance + img.getPixel(i, j));
            }
        }

        int ti = omp_get_thread_num();
        if(ti == 0)
            std::cout << progressbar((double)k, (double)N) << " " << percentage((double)k, (double)N) << "\r" << std::flush;
    }
    std::cout << std::endl;
    t.stop();

    img.divide((double)N);
    img.gamma_correlation();
    img.ppm_output("output.ppm");
}


Vec3 randVec() {
    return Vec3(2.0*noise()-1, 2.0*noise()-1, 2.0*noise()-1);
}


int main(void) {
    Vec3 sensor_center_pos = Vec3(0, 2, 6);
    Vec3 sensor_front = (-sensor_center_pos).normalize();
    Camera cam(sensor_center_pos, sensor_front, 1.0, 0.84, 2.0);
    cam.print();
    
    RGB royalBlue = RGB(65/255.0, 105/255.0, 225/255.0);
    Scene scene;
    scene.add(new Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), 10.0, Diffuse(RGB(1, 1, 1))));
    scene.add(new Sphere(Vec3(0), 1, Glass(RGB(1), 1.4)));
    scene.add(new Sphere(Vec3(0, 0, -5), 1.0, Diffuse(royalBlue)));
    scene.add(new Sphere(Vec3(-3, 0, 0), 1, Diffuse(RGB(0.9))));
    scene.add(new Sphere(Vec3(3, 0, 0), 1, Mirror(RGB(1))));
    
   
    /*
    scene.add(new Plane(Vec3(0, 9.9, 0), Vec3(0, -1, 0), 1.5, Emissive(RGB(10))));
    scene.add(new Plane(Vec3(-5, 5, 0), Vec3(1, 0, 0), 5.0, Diffuse(RGB(1, 0, 0))));
    scene.add(new Plane(Vec3(5, 5, 0), Vec3(-1, 0, 0), 5.0, Diffuse(RGB(0, 1, 0))));
    scene.add(new Plane(Vec3(0, 0, 0), Vec3(0, 1, 0), 5.0, Diffuse(RGB(1))));
    scene.add(new Plane(Vec3(0, 10, 0), Vec3(0, -1, 0), 5.0, Diffuse(RGB(1))));
    scene.add(new Plane(Vec3(0, 5, -5), Vec3(0, 0, 1), 5.0, Diffuse(RGB(1))));
    scene.add(new Sphere(Vec3(-2, 1.5, 0), 1.5, Mirror(RGB(1))));
    scene.add(new Sphere(Vec3(2, 1, 1.5), 1, Glass(RGB(1), 1.5)));
    */

    render_normal(cam, scene);
    render_depth(cam, scene);
    render(cam, scene);
}

