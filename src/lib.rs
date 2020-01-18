pub mod config {
    pub struct MapBounds {
        pub max_lon: f64,
        pub min_lon: f64,
        pub max_lat: f64,
        pub min_lat: f64
    }
}

pub mod read_map_data {
    use osmpbfreader::{Node, NodeId, Way};
    use petgraph::prelude::*;
    use petgraph::graph::*;
    use nannou::prelude::*;
    use std::collections::HashMap;
    use crate::config::MapBounds;

    pub fn road_graph_from_map_data(filepath: &str, map_bounds: &MapBounds) -> Graph<Node, f32, Directed> {
        let roads = read_road_data_from_map_file(filepath);
        let roads = exclude_roads_not_in_bounds(&roads, map_bounds);
        build_road_graph(roads)
    }

    /**
Reads the OSM PBF file using osmpbfreader crate, and returns lists of "roads" (which are lists of OSM Node objects).
*/
    fn read_road_data_from_map_file(filepath: &str) -> Vec<Vec<Node>> {
        let file = std::fs::File::open(&std::path::Path::new(filepath)).unwrap();
        let mut osm_file_reader = osmpbfreader::OsmPbfReader::new(file);
        let mut nodes = HashMap::new(); // associates OSM NodeIds with OSM Nodes
        let mut ways: Vec<Way> = Vec::new(); // OSM data stores roads as Way objects, which are lists of Node IDs

        for obj in osm_file_reader.par_iter().map(Result::unwrap) {
            match obj {
                osmpbfreader::OsmObj::Node(node) => {
                    nodes.insert(node.id, node);
                }
                osmpbfreader::OsmObj::Way(way) => {
                    if way.tags.contains_key("highway") {
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
Takes the list of Ways and then returns a new list with only those Ways that have at least one node in bounds.
*/
    fn exclude_roads_not_in_bounds(roads: &Vec<Vec<Node>>, map_bounds: &MapBounds) -> Vec<Vec<Node>> {
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
Given OSM Nodes, finds the geographical distance between (Pythagorean theorem).
*/
    fn dist_between_osm_nodes(node1: &Node, node2: &Node) -> f32 {
        let pt1 = pt2(node1.lon() as f32, node1.lat() as f32);
        let pt2 = pt2(node2.lon() as f32, node2.lat() as f32);
        ((pt1.x - pt2.x).powi(2) + (pt1.y - pt2.y).powi(2)).sqrt()
    }
}

