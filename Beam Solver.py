from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import matplotlib
import numpy as np
#import for writing csv

@dataclass(frozen=True)
class Material:
    name: str
    tensile: float
    compressive: float
    shear: float
    glue_shear: float
    density: float
    elastic_modulus: float

RED_OAK = Material(
    name="Red Oak",
    tensile=14300,
    compressive=6760,
    shear=1780,
    glue_shear=1391,
    density=0.026,
    elastic_modulus=1_800_000,
)

PINE = Material(
    name="Pine",
    tensile=12800,
    compressive=7000,
    shear=1390,
    glue_shear=989,
    density=0.018,
    elastic_modulus=1_500_000,
)


@dataclass
class Segment:
    material: Material
    width: float
    height: float
    y_bottom: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def y_bar(self) -> float:
        return self.y_bottom + (self.height / 2.0)

    @property
    def I_local(self) -> float:
        return (1.0 / 12.0) * self.width * (self.height ** 3)

@dataclass
class GlueJoint:
    name: str
    y_cut: float  # Vertical height where the glue cut occurs
    t_total: float  # Total contact width of the glue line(s)
    materials: Tuple[Material, Material]  # The two materials glued together
    # Calculated dynamically by CrossSection:
    # Maps ref_material.name -> transformed Q
    Q: Dict[str, float]

    @property
    def allowable_shear(self) -> float:
        """Glue strength is governed by the weaker of the glue shear ratings."""
        return min(self.materials[0].glue_shear, self.materials[1].glue_shear)

class CrossSection:
    def __init__(self, segments: List[Segment], joints: List[GlueJoint]):
        self.segments = segments
        self.joints = joints
        
        self.properties: Dict[str, dict] = {}
        self.y_bar: float = 0.0
        self.linear_density: float = 0.0
        self._calculate_properties()
    
    def _calculate_properties(self):
        total_AE = sum(segment.area * segment.material.elastic_modulus for segment in self.segments)
        first_moment_E = sum(segment.material.elastic_modulus * segment.area * segment.y_bar for segment in self.segments)
        
        self.y_bar = first_moment_E / total_AE
        
        self.linear_density = sum(segment.material.density * segment.area for segment in self.segments)

        unique_materials = {segment.material for segment in self.segments}
            
        for ref_material in unique_materials:
            ref_elastic_modulus = ref_material.elastic_modulus
            
            area = 0.0
            I = 0.0
            c_min = float("inf")
            c_max = float("-inf")
            
            for segment in self.segments:
                ratio = segment.material.elastic_modulus / ref_elastic_modulus
                
                area += segment.area * ratio
                
                moment = segment.I_local
                parallel_Axis = segment.area * (segment.y_bar - self.y_bar) ** 2
                
                I += (moment + parallel_Axis) * ratio
                
                if segment.material == ref_material:
                    y_bot_rel = segment.y_bottom - self.y_bar
                    y_top_rel = (segment.y_bottom + segment.height) - self.y_bar

                    c_min = min(c_min, y_bot_rel)
                    c_max = max(c_max, y_top_rel)
                
            
            self.properties[ref_material.name] = {
                "area": area,
                "I": I,
                "c_min": c_min,
                "c_max": c_max,
            }
    
    @classmethod
    def box_beam(
        cls,
        width: float,
        bottom_material: Material,
        bottom_height: float,
        
        web_material: Material,
        web_width: float,
        web_height: float,
        
        top_material: Material,
        top_height: float
        
    ) -> "CrossSection":
        segments = [
            #Bottom flange
            Segment(
                material=bottom_material,
                width=width,
                height=bottom_height,
                y_bottom=0 
            ),
            #Webs
            #TODO: isn't this just equilivant to an I beam in all calculations
            Segment(
                material=web_material,
                width=web_width,
                height=web_height,
                y_bottom=bottom_height
            ),
            Segment(
                material=web_material,
                width=web_width,
                height=web_height,
                y_bottom=bottom_height
            ),
            #Top Flange
            Segment(
                material=top_material,
                width=width,
                height=top_height,
                y_bottom=bottom_height+web_height
            )
        ]
        return cls(segments)

    @classmethod
    def I_beam(
        cls,
        bottom_material: Material,
        bottom_width: float,
        bottom_height: float,
        
        web_material: Material,
        web_width: float,
        web_height: float,
        
        top_material: Material,
        top_width: float,
        top_height: float,
    ) -> "CrossSection":
        segments = [
            #Bottom flange
            Segment(
                material=bottom_material,
                width=bottom_width,
                height=bottom_height,
                y_bottom=0 
            ),
            #Web
            Segment(
                material=web_material,
                width=web_width,
                height=web_height,
                y_bottom=bottom_height
            ),
            #Top Flange
            Segment(
                material=top_material,
                width=top_width,
                height=top_height,
                y_bottom=bottom_height+web_height
            )
        ]
        return cls(segments)
    
    @classmethod
    def T_beam(
        cls,
        web_material: Material,
        web_width: float,
        web_height: float,
        
        top_material: Material,
        top_width: float,
        top_height: float
    ) -> "CrossSection":
        segments = [
            #Web
            Segment(
                material=web_material,
                width=web_width,
                height=web_height,
                y_bottom=0
            ),
            #Top Flange
            Segment(
                material=top_material,
                width=top_width,
                height=top_height,
                y_bottom=web_height
            )
        ]
        return cls(segments)
    
    @classmethod
    def inverted_T_beam(
        cls,
        bottom_material: Material,
        bottom_width: float,
        bottom_height: float,
        
        web_material: Material,
        web_width: float,
        web_height: float
    ) -> "CrossSection":
        segments = [
            #Bottom flange
            Segment(
                material=bottom_material,
                width=bottom_width,
                height=bottom_height,
                y_bottom=0 
            ),
            #Web
            Segment(
                material=web_material,
                width=web_width,
                height=web_height,
                y_bottom=bottom_height
            )
        ]
        return cls(segments)