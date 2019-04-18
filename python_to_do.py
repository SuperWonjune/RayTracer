import os
import sys
from math import sqrt, pow, pi
import pdb  # use pdb.set_trace() for debugging
import code # or use code.interact(local=dict(globals(), **locals()))  for debugging.

import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image 


"""
Vector3 Class Based on the implementation idea of Vector from
https://gist.github.com/MartenMoti/7dacaff8e8f59d4aafac5560d66a089c
"""
class Vector3:
    def __init__(self,x,y,z):
        self.x, self.y, self.z = x, y, z
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y+other.y, self.z+other.z)
    def __sub__(self, other):
        return Vector3(self.x-other.x, self.y-other.y, self.z-other.z)
    def __mul__(self, other):
        return Vector3(self.x*other, self.y*other, self.z*other)
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
    def cross(self, other):
        return Vector3(self.y*other.z-self.z*other.y, self.z*other.x-self.x*other.z, self.x*other.y-self.y*other.x)
    def magnitude(self):
        return sqrt(self.x**2+self.y**2+self.z**2)
    def normal(self):
        mag = self.magnitude()
        return Vector3(self.x/mag,self.y/mag,self.z/mag)
    

class Light:
    def __init__(self, o:Vector3 ,intensity:float):
        self.o = o
        self.intensity = intensity


class Color:
    def __init__(self, rgb):
        # np.array float type
        self.color=rgb

    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma;
        self.color=np.power(self.color, inverseGamma)

    def toUINT8(self):
        return (np.clip(self.color, 0,1)*255).astype(np.uint8)


class Shader:
    """
    phong type은 diffuseColor, specularColor, exponent 소유
    Lambertian type은 diffuseColor만 소유

    color는 .2 .3 1과 같은 형태로 저장
    사용시 toUINT8() 메소드 활용
    """
    def __init__(self, name, type, diffuseColor:Color, specularColor:Color = Color(np.zeros(3)), exponent = 0):
        self.name = name
        self.type = type
        self.diffuseColor = diffuseColor
        self.specularColor = specularColor
        self.exponent = exponent


class Sphere:
    def __init__(self, center, radius, shader:Shader):
        self.center = center
        self.radius = radius
        self.shader = shader
        self.color = npToVector3(shader.diffuseColor.toUINT8())


    def intersection(self, ray):
        # 판별식 양, 음수 여부만 판별
        q = (ray.p - self.center).dot(ray.d)**2 - (ray.p - self.center).dot(ray.p - self.center) + self.radius**2
        if q < 0:
            # ray와 원이 안 만나는 경우
            return Intersection( Vector3(0,0,0), -1, Vector3(0,0,0), self)
        else:
            d = -ray.d.dot(ray.p - self.center)
            d1 = d - sqrt(q)
            d2 = d + sqrt(q)
            if 0 < d1 and ( d1 < d2 or d2 < 0):
                return Intersection(ray.p+ray.d*d1, d1, self.normal(ray.p+ray.d*d1), self)
            elif 0 < d2 and ( d2 < d1 or d1 < 0):
                return Intersection(ray.p+ray.d*d2, d2, self.normal(ray.p+ray.d*d2), self)
            else:
                return Intersection( Vector3(0,0,0), -1, Vector3(0,0,0), self)

    def normal(self, point):
        return (point - self.center).normal()


class Box:
    def __init__(self, minPt, maxPt, shader:Shader):
        self.minPt = minPt
        self.maxPt = maxPt
        self.shader = Shader
        self.color = npToVector3(shader.diffuseColor.toUINT8())

    def intersection(self, ray):
        return


class Ray:
    def __init__(self, eye, direction):
        self.p = eye
        self.d = direction

class Intersection:
    def __init__(self, point, distance, normal, obj):
        self.p = point
        self.d = distance
        self.n = normal
        self.obj = obj

def object_ray(ray, objects, ignore=None):
    intersect = Intersection( Vector3(0,0,0), -1, Vector3(0,0,0), None)

    for obj in objects:
        if obj is not ignore:
            currentIntersect = obj.intersection(ray)
            if currentIntersect.d > 0 and intersect.d < 0:
                intersect = currentIntersect
            elif 0 < currentIntersect.d < intersect.d:
                intersect = currentIntersect
    return intersect

def trace(ray, objects, light:Light):
    intersect = object_ray(ray, objects)

    # 공허
    if intersect.d == -1:
        col = Vector3(0, 0, 0)

    # 물체 음영
    elif intersect.n.dot(light.o - intersect.p) < 0:
        col = Vector3(0, 0, 0)
    else:
        lightRay = Ray(intersect.p, (light.o-intersect.p).normal())
        if object_ray(lightRay, objects, intersect.obj).d == -1:
            lightIntensity = 500.0/(4*pi*(light.o-intersect.p).magnitude()**2) * light.intensity
            col = intersect.obj.color * max(intersect.n.normal().dot((light.o - intersect.p).normal()*lightIntensity), AMBIENT)
        else:
            # 물체 위 그림자
            col = Vector3(0, 0, 0)
    return col

def gammaCorrection(color,factor):
    return (int(pow(color.x/255.0,factor)*255),
            int(pow(color.y/255.0,factor)*255),
            int(pow(color.z/255.0,factor)*255))


AMBIENT = 0.0
GAMMA_CORRECTION = 1/2.2

def getVecFromXml(root, elementName:str):
    arr = np.array(root.findtext(elementName).split()).astype(np.float)
    return Vector3(arr[0], arr[1], arr[2])


def npToVector3(np_arr):
    return Vector3(np_arr[0], np_arr[1], np_arr[2])


def main():
    # Get File Directory
    tree = ET.parse(sys.argv[1])
    # <scene> element in xml
    root = tree.getroot()

    # set default values
    viewDir=np.array([0,0,-1]).astype(np.float)
    viewUp=np.array([0,1,0]).astype(np.float)
    viewProjNormal=-1*viewDir  # you can safely assume this. (no examples will use shifted perspective camera)
    viewWidth=1.0
    viewHeight=1.0
    projDistance=1.0
    intensity=np.array([1,1,1]).astype(np.float)  # how bright the light is.

    """
    LOAD FROM XML BEGINS
    """
    
    imgSize=np.array(root.findtext('image').split()).astype(np.int)
    # imgSize[0]: width
    # imgSize[1]: height

    # Set values from <camera>
    for c in root.findall('camera'):
        viewPoint = getVecFromXml(c, 'viewPoint')
        viewDir = getVecFromXml(c, 'viewDir')
        viewUp = getVecFromXml(c, 'viewUp')
        viewProjNormal = getVecFromXml(c, 'projNormal')

        # single value
        viewHeight = float(c.findtext('viewHeight', default = viewHeight))
        viewWidth=float(c.findtext('viewWidth', default = viewWidth))
        projDistance=float(c.findtext('projDistance', default=projDistance))


    # Set values from <shader>
    # Shader Dictionary에 삽입
    # shaders[name: class Shader]
    # 넣을때 Color instance로 넣어버리자
    shaders = {}
    for c in root.findall('shader'):
        c_type = c.get('type')
        c_name = c.get('name')
        if c_type == "Phong":
            shaders[c_name] = Shader(c_name, c_type, 
                Color(np.array(c.findtext('diffuseColor').split()).astype(np.float)),
                Color(np.array(c.findtext('specularColor').split()).astype(np.float)),
                float(c.findtext('exponent')))
        elif c_type == "Lambertian":
            shaders[c_name] = Shader(c_name, c_type, 
                Color(np.array(c.findtext('diffuseColor').split()).astype(np.float)))
    print(shaders)


    # objects to be rendered. Find them in <surface> tag
    # 이 objects 배열은 추후 렌더링에 활용
    objects = []
    for c in root.findall('surface'):
        # Sphere
        if c.get('type') == 'Sphere':
            shade_name = c.find('shader').get('ref')
            objects.append(Sphere(npToVector3(np.array(c.findtext('center').split()).astype(np.float)),
                                float(c.findtext('radius')),
                                shaders[shade_name]))
        
        # box
        # TODO
        # elif c.get('type') == 'Box':

    # Set values from <light>
    for c in root.findall('light'):
        # array value
        lights = []
        light_position=np.array(c.findtext('position').split()).astype(np.float)
        intensity = np.array(c.findtext('intensity').split()).astype(np.float)[0]
        light_obj = Light(Vector3(light_position[0], light_position[1], light_position[2]), intensity)
        lights.append(light_obj)

    """
    LOAD FROM XML ENDS
    """

    # TODO
    # multiple lights
    lightSource = Light(Vector3(0,5,0), 1)

    img_width = imgSize[0]
    img_height = imgSize[1]

    img = Image.new("RGB",(img_width,img_height))
    cameraPos = viewPoint

    # get w,u,v Vectors
    d = viewDir
    w = d * -1
    u = w.cross(viewUp)
    u = u.normal()
    v = u.cross(w) * -1
    v = v.normal()

    for x in range(img_width):
        for y in range(img_height):
            d_norm = d.normal() * projDistance
            dirr_vec = d_norm + u*(-(viewWidth / 2) + (x / img_width) * viewWidth) * -1 + v*( (viewHeight / 2) - (y / img_height) * viewHeight)
            ray = Ray(cameraPos, dirr_vec.normal())
            col = trace(ray, objects, lightSource)
            img.putpixel((x,img_height-1-y),gammaCorrection(col,GAMMA_CORRECTION))
    img.save(sys.argv[1] + '.png')

    code.interact(local=dict(globals(), **locals()))

if __name__=="__main__":
    main()
