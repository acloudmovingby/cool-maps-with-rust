use map_project::nannou_conversions::{Polygon, batch_convert_coord};
use map_project::config::*;
use map_project::read_map_data::read_buildings_from_map_data;
use nannou::prelude::*;
use ordered_float::OrderedFloat;
use osmpbfreader::{Node, NodeId};
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::time::Instant;

const HUE_MAX: f32 = 0.55; // the nannou API for hsv colors takes an f32 between 0.0 and 1.0
const HUE_MIN: f32 = 0.0;

const BACKGROUND_HUE: f32 = 0.73;
const BACKGROUND_SATURATION: f32 = 0.2;
const BACKGROUND_VALUE: f32 = 0.2;

fn main() {
    nannou::app(model).update(update).run();
}

fn setup_config() -> MapConfigData {
    let map_bounds = MapBounds {
        max_lon: -71.3919,
        min_lon: -71.4311,
        max_lat: 41.8300,
        min_lat: 41.8122,
    };
    let window_dimensions = calculate_window_dimensions(657.0,&map_bounds);
    let map_file_path = "/Users/christopherpoates/Downloads/rhode-island-latest.osm.pbf".to_string(); // RI
    //let map_file_path = "/Users/christopherpoates/Downloads/massachusetts-latest.osm.pbf".to_string(); // MA
    MapConfigData {map_bounds, window_dimensions, map_file_path}
}

fn calculate_window_dimensions(max_win_height: f32, map_bounds: &MapBounds) -> WindowDimensions {
    let lon_to_lat_ratio = ((map_bounds.max_lon-map_bounds.min_lon)/(map_bounds.max_lat-map_bounds.min_lat)) as f32;
    WindowDimensions{width: lon_to_lat_ratio*max_win_height, height: max_win_height}
}

struct Model {
    _window: window::Id,
    buildings: Vec<Polygon>, // each Vec represents a closed path of points describing the perimeter of the building
}

fn model(app: &App) -> Model {
    let config = setup_config();
    let _window = app
        .new_window()
        .with_dimensions(config.window_dimensions.width as u32, config.window_dimensions.height as u32)
        .view(view)
        .event(window_event)
        .build()
        .unwrap();
    let t1 = Instant::now();

    println!("t1 before reading map data {}", t1.elapsed().as_secs());

    let buildings = read_buildings_from_map_data(&config);
    let buildings = batch_convert_coord(&buildings, &config);

    println!("t1 after making the buildings: {}", t1.elapsed().as_secs());

    // create tuple pairs of buildings and their areas
    let building_areas: Vec<(Vec<Point2>, f32)> = calculate_building_areas(&buildings);
    let building_hues: Vec<(Vec<Point2>, f32)> = calculate_building_hues(&building_areas);
    let buildings = convert_to_polygons(&building_hues);

    Model {
        _window,
        buildings,
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn window_event(_app: &App, _model: &mut Model, event: WindowEvent) {
    match event {
        KeyPressed(_key) => {}
        KeyReleased(_key) => {}
        MouseMoved(_pos) => {}
        MousePressed(_button) => {}
        MouseReleased(_button) => {}
        MouseEntered => {}
        MouseExited => {}
        MouseWheel(_amount, _phase) => {}
        Moved(_pos) => {}
        Resized(_size) => {}
        Touch(_touch) => {}
        TouchPressure(_pressure) => {}
        HoveredFile(_path) => {}
        DroppedFile(_path) => {}
        HoveredFileCancelled => {}
        Focused => {}
        Unfocused => {}
        Closed => {}
    }
}

fn view(app: &App, model: &Model, frame: Frame) -> Frame {
    // Prepare to draw.
    let draw = app.draw();

    for building_polygon in &model.buildings {
        draw.polygon().points(building_polygon.points.clone()).hsv(building_polygon.hue, building_polygon.saturation, building_polygon.value);
    }

    draw.background().hsv(BACKGROUND_HUE, BACKGROUND_SATURATION, BACKGROUND_VALUE);
    // Write to the window frame.
    draw.to_frame(app, &frame).unwrap();
    // Return the drawn frame.
    frame
}




// for performance reasons you can use this to remove buildings below a certain pixel size (both the width/height of the building must exceed the min_size)
fn level_of_detail(min_size: f32, points: Vec<Vec<Point2>>) -> Vec<Vec<Point2>> {
    points
        .into_iter()
        .filter(|pts| {
            let mut min_y = std::f32::MAX;
            let mut max_y = -std::f32::MAX;
            let mut min_x = std::f32::MAX;
            let mut max_x = -std::f32::MAX;
            for pt in pts {
                if pt.x > max_x {
                    max_x = pt.x;
                }
                if pt.x < min_x {
                    min_x = pt.x;
                }
                if pt.y > max_y {
                    max_y = pt.y;
                }
                if pt.y < min_y {
                    min_y = pt.y;
                }
            }
            (max_x - min_x > min_size) && (max_y - min_y > min_size)
        })
        .collect()
}

fn polygon_area(points: &[Point2<f32>]) -> f32 {
    if points.is_empty() {
        0.0
    } else {
        let mut area = 0.0;
        let mut prev = points.iter().last().unwrap();

        for point in points {
            area += (point.x + prev.x) * (point.y - prev.y);
            prev = point;
        }
        (area / 2.0).abs()
    }
}

fn calculate_building_areas(buildings: &Vec<Vec<Point2>>) -> Vec<(Vec<Point2>, f32)> {
    buildings
        .into_iter()
        .map(|points| {
            let area = polygon_area(&points).min(500.0);
            (points.clone(), area)
        })
        .collect()
}

fn calculate_building_hues(buildings: &Vec<(Vec<Point2>, f32)>) -> Vec<(Vec<Point2>, f32)> {
    let min_area = buildings
        .iter()
        .map(|(_pts, area)| OrderedFloat(*area))
        .min()
        .unwrap()
        .0;
    let max_area = buildings
        .iter()
        .map(|(_pts, area)| OrderedFloat(*area))
        .max()
        .unwrap()
        .0;
    buildings
        .iter()
        .map(|(pts, area)| {
            let foo = pts.clone();
            let hue = map_range(*area, min_area, max_area, HUE_MIN, HUE_MAX);
            (pts.clone(), hue)
        })
        .collect()
}

fn convert_to_polygons(buildings: &Vec<(Vec<Point2>, f32)>) -> Vec<Polygon> {
    buildings
        .iter()
        .map(|(points, hue)| Polygon {
            points: points.clone(),
            hue: *hue,
            saturation: 1.0,
            value: 0.5,
        })
        .collect()
}
