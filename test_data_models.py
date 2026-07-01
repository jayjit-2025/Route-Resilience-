"""Simple validation script for core data models.

This script verifies that all dataclasses are properly defined with
correct type annotations and can be instantiated.
"""

import sys
from dataclasses import fields


def validate_dataclass(cls, expected_fields):
    """Validate that a dataclass has expected fields with proper types."""
    class_fields = {f.name: f.type for f in fields(cls)}
    
    print(f"\n✓ {cls.__name__}:")
    print(f"  - Has @dataclass decorator: {hasattr(cls, '__dataclass_fields__')}")
    print(f"  - Number of fields: {len(class_fields)}")
    print(f"  - Has docstring: {cls.__doc__ is not None}")
    
    for field_name in expected_fields:
        if field_name in class_fields:
            print(f"  - Field '{field_name}': {class_fields[field_name]}")
        else:
            print(f"  - Missing field: '{field_name}'")
            return False
    
    return True


if __name__ == "__main__":
    # We need to mock the imports that aren't installed yet
    import types
    
    # Mock modules
    sys.modules['numpy'] = types.ModuleType('numpy')
    sys.modules['networkx'] = types.ModuleType('networkx')
    sys.modules['rasterio'] = types.ModuleType('rasterio')
    sys.modules['rasterio.transform'] = types.ModuleType('rasterio.transform')
    sys.modules['rasterio.crs'] = types.ModuleType('rasterio.crs')
    
    # Add mock classes
    class MockCRS:
        pass
    class MockAffine:
        pass
    class MockNDArray:
        pass
    class MockGraph:
        pass
    
    sys.modules['rasterio.crs'].CRS = MockCRS
    sys.modules['rasterio.transform'].Affine = MockAffine
    sys.modules['numpy'].ndarray = MockNDArray
    sys.modules['networkx'].Graph = MockGraph
    
    # Now import our data models
    from core.data_models import (
        GeoMetadata, 
        PipelineState, 
        CentralityResult, 
        SimulationMetrics, 
        RenderConfig
    )
    
    print("=" * 60)
    print("VALIDATING CORE DATA MODELS")
    print("=" * 60)
    
    # Validate each dataclass
    all_valid = True
    
    all_valid &= validate_dataclass(
        GeoMetadata,
        ['crs', 'transform', 'bounds', 'shape']
    )
    
    all_valid &= validate_dataclass(
        PipelineState,
        ['raw_image', 'geo_metadata', 'preprocessed_image', 'road_mask', 
         'skeleton', 'junction_coords', 'endpoint_coords', 'raw_graph',
         'healed_graph', 'connectivity_ratio', 'centrality_result',
         'simulation_metrics', 'modified_graph']
    )
    
    all_valid &= validate_dataclass(
        CentralityResult,
        ['node_centrality', 'gatekeeper_nodes', 'threshold_value']
    )
    
    all_valid &= validate_dataclass(
        SimulationMetrics,
        ['travel_delay', 'components', 'efficiency', 'resilience']
    )
    
    all_valid &= validate_dataclass(
        RenderConfig,
        ['tile_layer', 'zoom_start', 'road_color', 'gatekeeper_color', 
         'heatmap_gradient']
    )
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ ALL DATACLASSES VALIDATED SUCCESSFULLY")
    else:
        print("✗ VALIDATION FAILED")
    print("=" * 60)
    
    # Test instantiation
    print("\nTesting instantiation with defaults...")
    try:
        # Test PipelineState (has defaults)
        state = PipelineState()
        print("✓ PipelineState() - Created with defaults")
        
        # Test RenderConfig (has defaults)
        config = RenderConfig()
        print(f"✓ RenderConfig() - Created with defaults (zoom_start={config.zoom_start})")
        
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Instantiation failed: {e}")
        sys.exit(1)
