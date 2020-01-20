use nannou::prelude::*;
use osmpbfreader::{Node};
use petgraph::graph::*;
use petgraph::prelude::*;
use map_project::read_map_data::read_buildings_from_map_data;
use map_project::read_map_data::road_graph_from_map_data;
use map_project::config::{MapConfigData, MapBounds, WindowDimensions};
use map_project::nannou_conversions::{convert_coord, batch_convert_coord, Line, Polygon};

fn main() {
    nannou::app(model).run();
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
    road_lines: Vec<Line>,       // Line stores start, end, hue, saturation, alpha, thickness
}

/**
Sets up the initial state of the program before the nannou UI loop begins.
*/
fn model(app: &App) -> Model {
    let config = setup_config();
    let _window = app
        .new_window()
        .with_dimensions(config.window_dimensions.width as u32, config.window_dimensions.height as u32)
        .view(view)
        .event(window_event)
        .build()
        .unwrap();

    let road_graph = road_graph_from_map_data(&config);
    let road_lines: Vec<Line> = color_roads(&road_graph, &config);
    let buildings = read_buildings_from_map_data(&config);
    let buildings: Vec<Vec<Point2>> = batch_convert_coord(&buildings, &config);
    let buildings: Vec<Polygon> = convert_to_polygon_objs(&buildings);

    Model {
        _window,
        buildings,
        road_lines,
    }
}

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

/**
Nannou runs function 60 times per second to produce a new frame.
*/
fn view(app: &App, model: &Model, frame: Frame) -> Frame {
    let _win = app.window_rect();
    let draw = app.draw();

    for building_polygon in model.buildings.iter() {
        draw.polygon().points(building_polygon.points.clone()).hsv(building_polygon.hue, building_polygon.saturation, building_polygon.value);
    }
    /*
    for building in &model.buildings {
        let mut points: Vec<Point2> = Vec::new();
        for node in building {
            points.push(node.clone());
        }
        draw.polygon().points(points).hsv(0.6, 0.7, 0.5);
    }*/

    for road_line in model.road_lines.iter() {
        draw.line()
            .points(road_line.start, road_line.end)
            .thickness(road_line.thickness)
            .hsva(road_line.hue, road_line.saturation, 1.0, road_line.alpha);
    }

    draw.background().hsv(0.85, 0.3, 0.65);
    // Write to the window frame.
    draw.to_frame(app, &frame).unwrap();
    // Return the drawn frame.
    frame
}


/**
Currently not used. Min_size is the minimum pixel width a building has to span in order to be drawn.
This function returns the list of building points with any buildings smaller than the minimum removed.
*/
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

/**
Given the road graph, it transforms it into Lines to be fed to the nannou drawing functions
*/
fn color_roads(road_graph: &Graph<Node, f32, Directed>, config: &MapConfigData) -> Vec<Line> {
    let mut road_lines = Vec::new();
    for edge in road_graph.raw_edges() {
        let source = road_graph
            .node_weight(edge.source())
            .map(|node| convert_coord(node,config));
        let target = road_graph
            .node_weight(edge.target())
            .map(|node| convert_coord(node,config));

        if source.is_some() && target.is_some() {
            road_lines.push(Line {
                start: source.unwrap(),
                end: target.unwrap(),
                thickness: 2.0,
                hue: 1.0,
                saturation: 0.0,
                alpha: 1.0,
            });
        }
    }
    road_lines
}

fn convert_to_polygon_objs(node_lists: &Vec<Vec<Point2>>) -> Vec<Polygon> {
    node_lists.iter()
        .map(|points| Polygon{
            points: points.clone(),
            hue: 0.6,
            saturation: 0.7,
            value: 0.5
        })
        .collect()
}
