pub mod config {
    /**
    Data necessary for configuring how to draw the maps (these are used by all the binaries in map_project)
    */
    pub struct MapConfigData {
        pub map_bounds: MapBounds,
        pub window_dimensions: WindowDimensions,
        pub map_file_path: String
    }

    pub struct MapBounds {
        pub max_lon: f64,
        pub min_lon: f64,
        pub max_lat: f64,
        pub min_lat: f64
    }

    pub struct WindowDimensions {
        pub width: f32,
        pub height: f32
    }
}

pub mod read_map_data {
    use osmpbfreader::{Node, NodeId, Way};
    use petgraph::prelude::*;
    use petgraph::graph::*;
    use nannou::prelude::*;
    use std::collections::HashMap;
    use crate::config::{MapBounds, MapConfigData};
    use std::collections::hash_map::RandomState;

    pub fn read_buildings_from_map_data(config: &MapConfigData) -> Vec<Vec<Node>> {
        let buildings = get_ways_from_map_file("building", &config.map_file_path);
        exclude_not_in_bounds(&buildings, &config.map_bounds)
    }

    pub fn road_graph_from_map_data(config: &MapConfigData) -> Graph<Node, f32, Directed> {
        let roads = get_ways_from_map_file("highway", &config.map_file_path);
        let roads = exclude_not_in_bounds(&roads, &config.map_bounds);
        build_road_graph(roads)
    }

    fn get_ways_from_map_file(way_tag: &str, map_file_path:&str) -> Vec<Vec<Node>> {
        let file = std::fs::File::open(&std::path::Path::new(map_file_path)).unwrap();
        let mut osm_file_reader = osmpbfreader::OsmPbfReader::new(file);
        let mut ways = Vec::new();
        let mut nodes = HashMap::new(); // associates OSM NodeIds with OSM Nodes

        for obj in osm_file_reader.par_iter().map(Result::unwrap) {
            match obj {
                osmpbfreader::OsmObj::Node(node) => {
                    nodes.insert(node.id, node);
                }
                osmpbfreader::OsmObj::Way(way) => {
                    if way.tags.contains_key(way_tag) {
                        ways.push(way);
                    }
                }
                osmpbfreader::OsmObj::Relation(_rel) => {}
            }
        }
        convert_ways_into_node_vecs(&nodes, &ways)
    }

    /**
OSM Way objects only store lists of NodeIds, not the Node objects themselves, so to simplify things this function converts all Way objects into lists of Node objects (so you don't need to keep passing around a hashmap associating NodeIds with Nodes)
*/
    fn convert_ways_into_node_vecs(nodes: &HashMap<NodeId, Node>, roads: &Vec<Way>) -> Vec<Vec<Node>> {
        roads
            .iter()
            .map(|way| {
                way.nodes
                    .iter()
                    .map(|node_id| nodes.get(node_id).unwrap().clone())
                    .collect()
            })
            .collect()
    }

    /**
Takes the list of Ways (buildings, roads, etc.) that have been converted into lists of nodes (Vec<Node>), and then returns only those of them that have at least one node in bounds. (thus excluding all non-visible roads/buildings)
*/
    pub fn exclude_not_in_bounds(roads: &Vec<Vec<Node>>, map_bounds: &MapBounds) -> Vec<Vec<Node>> {
        let mut roads_in_bounds: Vec<Vec<Node>> = Vec::new();
        for road in roads.iter() {
            let any_in_bounds = road.iter().any(|node| is_in_bounds(node, map_bounds));
            // if any of the nodes in the road are in bounds, then keep the road for the graph
            if any_in_bounds {
                roads_in_bounds.push(road.clone());
            }
        }
        roads_in_bounds
    }

    fn build_road_graph(roads: Vec<Vec<Node>>) -> Graph<Node, f32, Directed> {
        let mut graph_node_indices = HashMap::new();
        let mut road_graph = Graph::<Node, f32, Directed>::new();
        for road in &roads {
            let mut prior_node_index = NodeIndex::new(0);
            for (i, node) in road.iter().enumerate() {
                let cur_node_index: NodeIndex;
                if graph_node_indices.contains_key(node) {
                    cur_node_index = *graph_node_indices.get(node).unwrap();
                } else {
                    cur_node_index = road_graph.add_node(node.clone());
                    graph_node_indices.insert(node, cur_node_index);
                }

                // if it's not the first one, form an edge along the path
                if i != 0 {
                    let prior_node = road_graph
                        .node_weight(prior_node_index)
                        .expect("prior node should exist because we already traversed it");
                    let dist = dist_between_osm_nodes(&node, &prior_node);
                    road_graph.add_edge(prior_node_index, cur_node_index, dist);
                    road_graph.add_edge(cur_node_index, prior_node_index, dist);
                }
                prior_node_index = cur_node_index;
            }
        }
        road_graph
    }

    fn is_in_bounds(node: &Node, map_bounds: &MapBounds) -> bool {
        (node.lon() < map_bounds.max_lon)
            & (node.lon() > map_bounds.min_lon)
            & (node.lat() < map_bounds.max_lat)
            & (node.lat() > map_bounds.min_lat)
    }


    /**
Given OSM Nodes, finds the geographical distance between them (Pythagorean theorem).
*/
    fn dist_between_osm_nodes(node1: &Node, node2: &Node) -> f32 {
        let pt1 = pt2(node1.lon() as f32, node1.lat() as f32);
        let pt2 = pt2(node2.lon() as f32, node2.lat() as f32);
        ((pt1.x - pt2.x).powi(2) + (pt1.y - pt2.y).powi(2)).sqrt()
    }
}

/**
Responsible for useful functions to interact with the nannou API
*/
pub mod nannou_conversions {
    use crate::config::{MapBounds, MapConfigData};
    use nannou::prelude::*;
    use osmpbfreader::Node;

    /**
Converts the geographical coordinates of an OSM node (its longitudue/latitude) and converts it into a pixel coordinate to feed to the nannou drawing functions.
*/
    pub fn convert_coord(node: &Node, config: &MapConfigData) -> Point2 {
        // note that nannou draws to the screen with (0,0) is the center of the window, with negatives to the left, positives to the right
        let x = map_range(
            node.lon(),
            config.map_bounds.min_lon,
            config.map_bounds.max_lon,
            -config.window_dimensions.width * 0.5,
            config.window_dimensions.width * 0.5,
        );
        let y = map_range(
            node.lat(),
            config.map_bounds.min_lat,
            config.map_bounds.max_lat,
            -config.window_dimensions.height * 0.5,
            config.window_dimensions.height * 0.5,
        );
        pt2(x, y)
    }

    /**
    Same as convert_coord above, just for a list of lists of nodes (where each list of nodes represents some Way from the OSM data, like a building perimeter or a road).
    */
    pub fn batch_convert_coord(nodes: &Vec<Vec<Node>>, config: &MapConfigData) -> Vec<Vec<Point2>> {
        nodes.iter()
            .map(|node_list| node_list.iter()
                .map(|node| convert_coord(node, &config))
                .collect())
            .collect()
    }

    /**
Stores the data I want to use when I draw a line using nannou's draw.line() builder, namely the end points and values for the color, thickness, etc. This ensures that these values are all ready to go and there's no unnecessary calculations being formed in the nannou view function (which is called many times per second to build the frames)
*/
    pub struct Line {
        pub start: Point2<f32>,
        pub end: Point2<f32>,
        pub thickness: f32,
        pub hue: f32,
        pub saturation: f32,
        pub alpha: f32,
    }

    pub struct Polygon {
        pub points: Vec<Point2>,
        pub hue: f32,
        pub saturation: f32,
        pub value: f32
    }
}
