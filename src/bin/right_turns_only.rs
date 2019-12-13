use nannou::draw::properties::Vertices;
use nannou::prelude::*;
use ordered_float::OrderedFloat;
use osmpbfreader::{Node, NodeId, Way};
use petgraph::graph::NodeIndex;
use petgraph::graph::*;
use petgraph::prelude::*;
use petgraph::visit::{IntoEdges, IntoNeighbors};
use petgraph::algo::is_isomorphic_matching;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;

const UTURN_PENALTY: f32 = 0.0;
const LEFT_TURN_PENALTY: f32 = 0.0;
const RIGHT_TURN_PENALTY: f32 = 0.0;
const STRAIGHT_PENALTY: f32 = 0.0;



// AROUND MY HOUSE
/*
const MAX_LON: f64 = -71.10030;
const MIN_LON: f64 = -71.11010;
const MAX_LAT: f64 = 42.39705;
const MIN_LAT: f64 = 42.39263;*/

// DOWNTOWN BOSTON + CAMBRIDGE
/*
const MAX_LON: f64 = -71.0479;
const MIN_LON: f64 = -71.0871;
const MAX_LAT: f64 = 42.3704;
const MIN_LAT: f64 = 42.3527;*/

// NAHANT
/*
const MAX_LON: f64 = -70.8440;
const MIN_LON: f64 = -71.0009;
const MAX_LAT: f64 = 42.4808;
const MIN_LAT: f64 = 42.4070;*/

// NEWPORT / JAMESTOWN, RI
/*
const MAX_LON: f64 = -71.1856;
const MIN_LON: f64 = -71.4994;
const MAX_LAT: f64 = 41.5756;
const MIN_LAT: f64 = 41.4257;*/

// Near NEWPORT RI
/*
const MAX_LON: f64 = -71.3048;
const MIN_LON: f64 = -71.4133;
const MAX_LAT: f64 = 41.5108;
const MIN_LAT: f64 = 41.4749;*/

// PROVIDENCE DOWNTOWN
/*
const MAX_LON: f64 = -71.3919;
const MIN_LON: f64 = -71.4311;
const MAX_LAT: f64 = 41.8300;
const MIN_LAT: f64 = 41.8122;*/

// LARGER PROVIDENCE AREA
/*
const MAX_LON: f64 = -71.2202;
const MIN_LON: f64 = -71.5340;
const MAX_LAT: f64 = 41.8831;
const MIN_LAT: f64 = 41.7403;*/

/*
const MAX_LON: f64 = -71.36522;
const MIN_LON: f64 = -71.38602;
const MAX_LAT: f64 = 41.53705;
const MIN_LAT: f64 = 41.52632;*/

//Small part of Jamestown, RI
/*
const MAX_LON: f64 = -71.3621;
const MIN_LON: f64 = -71.3820;
const MAX_LAT: f64 = 41.5028;
const MIN_LAT: f64 = 41.4938;*/

const MAX_LON: f64 = 30.0;
const MIN_LON: f64 = -30.0;
const MAX_LAT: f64 = 30.0;
const MIN_LAT: f64 = -30.0;

// VERY small part of Jamestown, RI
/*
const MAX_LON: f64 = -71.36557;
const MIN_LON: f64 = -71.37553;
const MAX_LAT: f64 = 41.50079;
const MIN_LAT: f64 = 41.49631;*/

// random part or RI
/*
const MAX_LON: f64 = -71.2896;
const MIN_LON: f64 = -71.3095;
const MAX_LAT: f64 = 41.5251;
const MIN_LAT: f64 = 41.5162;*/

//Brown University
/*const MAX_LON: f64 = -71.3909;
const MIN_LON: f64 = -71.4105;
const MAX_LAT: f64 = 41.8282;
const MIN_LAT: f64 = 41.8192;*/

// Brown Frat squad
/*
const MAX_LON: f64 = -71.39901;
const MIN_LON: f64 = -71.40392;
const MAX_LAT: f64 = 41.82546;
const MIN_LAT: f64 = 41.82323;*/

const LON_RANGE: f64 = MAX_LON - MIN_LON;
const LAT_RANGE: f64 = MAX_LAT - MIN_LAT;

const MAX_WIN_WIDTH: f32 = 1357.0 * 0.7; // based on my laptop screen
const MAX_WIN_HEIGHT: f32 = 657.0 * 0.7; // based on my laptop screen

const WIN_W: f32 = ((MAX_WIN_HEIGHT as f64) / LAT_RANGE * LON_RANGE) as f32;
const WIN_H: f32 = MAX_WIN_HEIGHT;

const HUE_MAX: f32 = 0.9; // the nannou API for hsv colors takes an f32 between 0.0 and 1.0
const HUE_MIN: f32 = 0.55;

fn main() {
    nannou::app(model).run();
}

struct Model {
    _window: window::Id,
    road_lines: Vec<Line>, // Line stores start, end, hue, saturation, alpha, thickness
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .with_dimensions(WIN_W as u32, WIN_H as u32)
        .view(view)
        .event(window_event)
        .build()
        .unwrap();

    let mut start_node = NodeIndex::new(0);
    /*
        let filename = "/Users/christopherpoates/Downloads/rhode-island-latest.osm.pbf"; // RI
                                                                                         //let filename = "/Users/christopherpoates/Downloads/massachusetts-latest.osm.pbf"; // MA

        let r = std::fs::File::open(&std::path::Path::new(filename)).unwrap();
        let mut pbf = osmpbfreader::OsmPbfReader::new(r);

        let mut nodes = HashMap::new();
        let mut all_roads: Vec<Way> = Vec::new();

        // READING MAP DATA
        for obj in pbf.par_iter().map(Result::unwrap) {
            match obj {
                osmpbfreader::OsmObj::Node(node) => {
                    if is_in_outer_bounds(&node) {
                        nodes.insert(node.id, node);
                    }
                }
                osmpbfreader::OsmObj::Way(way) => {
                    if way.tags.contains_key("highway") {
                        all_roads.push(way);
                    }
                }
                osmpbfreader::OsmObj::Relation(_rel) => {}
            }
        }

        // now we take all_roads and remove all Ways that are not in map bounds to make a new collection: roads
        // roads thus represents all the Ways that have at least one node in the map bounds
        let mut roads: Vec<Way> = Vec::new();
        for road in all_roads {
            let any_in_bounds = road.nodes.iter().any(|node_id| {
                if nodes.contains_key(node_id) {
                    let node = nodes.get(node_id).unwrap();
                    is_in_bounds(node)
                } else {
                    false
                }
            });

            // if any of the nodes in the road are in bounds, then keep the road for the graph
            if any_in_bounds {
                roads.push(road);
            }
        }

        // BUILD GRAPH
        // make hashmap of node ids to node indices
        // before adding a node, check to see if it exists
        // when you add an edge, use node index from hashmap
        let mut graph_node_indices = HashMap::new();
        let mut road_graph = Graph::<Node, f32, Directed>::new();
        for road in &roads {
            let mut prior_node_index = NodeIndex::new(0);
            for (i, node_id) in road.nodes.iter().enumerate() {
                // look up node using id
                if let Some(node) = nodes.get(node_id) {
                    let cur_node_index: NodeIndex;
                    if graph_node_indices.contains_key(node) {
                        cur_node_index = *graph_node_indices.get(node).unwrap();
                    } else {
                        cur_node_index = road_graph.add_node(node.clone());
                        if (node.id.0 == 5955756766) {
                            println!("Found node 5955756766");
                            start_node = cur_node_index;
                        }
                        graph_node_indices.insert(node, cur_node_index);
                    }

                    // if it's not the first one, form an edge along the path
                    if i != 0 {
                        // find distances between the two points
                        let prior_node = road_graph
                            .node_weight(prior_node_index)
                            .expect("prior node should exist because we already traversed it");
                        let start_point = pt2(prior_node.lon() as f32, prior_node.lat() as f32);
                        let end_point = pt2(node.lon() as f32, node.lat() as f32);

                        // for any of those nodes, print the edge that is being added, so i can be sure it's adding those edges and no more
                        // for each node then, I should see two edges and those edge should be to the correct other points

                        road_graph.add_edge(
                            prior_node_index,
                            cur_node_index,
                            dist(start_point, end_point),
                        );
                        road_graph.add_edge(
                            cur_node_index,
                            prior_node_index,
                            dist(start_point, end_point),
                        );
                    }
                    prior_node_index = cur_node_index;
                }
            }
        }
    */
    let node0: Node = Node {
        id: NodeId(0),
        tags: Default::default(),
        decimicro_lat: 0,
        decimicro_lon: 0,
    };
    let node1: Node = Node {
        id: NodeId(1),
        tags: Default::default(),
        decimicro_lat: -10,
        decimicro_lon: 1,
    };
    let node2: Node = Node {
        id: NodeId(2),
        tags: Default::default(),
        decimicro_lat: 1,
        decimicro_lon: 10,
    };
    let node3: Node = Node {
        id: NodeId(3),
        tags: Default::default(),
        decimicro_lat: 10,
        decimicro_lon: -1,
    };
    let node4: Node = Node {
        id: NodeId(4),
        tags: Default::default(),
        decimicro_lat: -1,
        decimicro_lon: -10,
    };
    let node5: Node = Node {
        id: NodeId(5),
        tags: Default::default(),
        decimicro_lat: 10,
        decimicro_lon: 10,
    };
    let node6: Node = Node {
        id: NodeId(6),
        tags: Default::default(),
        decimicro_lat: 20,
        decimicro_lon: -1,
    };
    let node7: Node = Node {
        id: NodeId(7),
        tags: Default::default(),
        decimicro_lat: 10,
        decimicro_lon: -10,
    };
    let node8: Node = Node {
        id: NodeId(8),
        tags: Default::default(),
        decimicro_lat: 0,
        decimicro_lon: -10,
    };
    let node9: Node = Node {
        id: NodeId(9),
        tags: Default::default(),
        decimicro_lat: 10,
        decimicro_lon: -20,
    };
    let node10: Node = Node {
        id: NodeId(10),
        tags: Default::default(),
        decimicro_lat: 20,
        decimicro_lon: -10,
    };
    let mut road_graph = Graph::<Node, f32, Directed>::new();

    let n0 = road_graph.add_node(node0);
    let n1 = road_graph.add_node(node1);
    let n2 = road_graph.add_node(node2);
    let n3 = road_graph.add_node(node3);
    let n4 = road_graph.add_node(node4);
    let n5 = road_graph.add_node(node5);
    let n6 = road_graph.add_node(node6);
    let n7 = road_graph.add_node(node7);
    // let n8 = road_graph.add_node(node8);
    // let n9 = road_graph.add_node(node9);
    // let n10 = road_graph.add_node(node10);
    road_graph.add_edge(n0, n1, 1.0); // the edge weights don't actually matter here as we're only testing the structure of the graph
    road_graph.add_edge(n1, n0, 1.0);
    road_graph.add_edge(n0, n2, 1.0);
    road_graph.add_edge(n2, n0, 1.0);
    road_graph.add_edge(n0, n3, 1.0);
    road_graph.add_edge(n3, n0, 1.0);
    road_graph.add_edge(n0, n4, 1.0);
    road_graph.add_edge(n4, n0, 1.0);
    road_graph.add_edge(n3, n5, 1.0);
    road_graph.add_edge(n5, n3, 1.0);
    road_graph.add_edge(n3, n6, 1.0);
    road_graph.add_edge(n6, n3, 1.0);
    road_graph.add_edge(n3, n7, 1.0);
    road_graph.add_edge(n7, n3, 1.0);
    /* road_graph.add_edge(n7, n8, 1.0);
    road_graph.add_edge(n8, n7, 1.0);
    road_graph.add_edge(n7, n9, 1.0);
    road_graph.add_edge(n9, n7, 1.0);
    road_graph.add_edge(n7, n10, 1.0);
    road_graph.add_edge(n10, n7, 1.0);*/
    for edge in road_graph.edge_indices() {
        let (start, end) = road_graph.edge_endpoints(edge).unwrap();
        let start_weight = road_graph.node_weight(start).unwrap();
        let end_weight = road_graph.node_weight(end).unwrap();
        println!(
            "({},{} -> {},{})",
            start_weight.id.0,
            start.index(),
            end_weight.id.0,
            end.index()
        );
        // println!("({},{})",start_weight.id.0, end_weight.id.0);
    }

    let no_left_turns_graph = prohibit_left_turns2(&road_graph);

    println!("CHECKING GRAPH START");
    for edge in no_left_turns_graph.edge_indices() {
        let (start, end) = no_left_turns_graph.edge_endpoints(edge).unwrap();
        let start_weight = no_left_turns_graph.node_weight(start).unwrap();
        let end_weight = no_left_turns_graph.node_weight(end).unwrap();
        println!(
            "({},{} -> {},{})",
            start_weight.id.0,
            start.index(),
            end_weight.id.0,
            end.index()
        );
        // println!("({},{})",start_weight.id.0, end_weight.id.0);
    }
    println!("CHECKING GRAPH END");

    // find the minimum distance to every node in the graphs (both the original and the turn-modified graph)
    let orig_min_dist_to_nodes = djikstra_float(&road_graph, start_node);
    // find the index of the start node in the turn-modified graph, then find the minimum distance to very node in that turn-modified graph
    let start_osm_id = road_graph.node_weight(start_node).unwrap();
    let start_node = no_left_turns_graph
        .node_indices()
        .find(|&node_ix| start_osm_id == no_left_turns_graph.node_weight(node_ix).unwrap())
        .expect("no_left_turns_graph should have a node with that osm id.");
    let modified_min_dist_to_nodes = djikstra_float(&no_left_turns_graph, start_node);
    for (node_ix, min_dist) in modified_min_dist_to_nodes.iter() {
        let id = no_left_turns_graph.node_weight(*node_ix).unwrap().id.0;
        if min_dist.is_none() {
            println!("id: {}, ix: {}, min_dist: NONE", id, node_ix.index());
        } else {
            println!(
                "id: {}, ix: {}, min_dist: {}",
                id,
                node_ix.index(),
                min_dist.unwrap()
            );
        }
    }

    // now make graphs where the edges themselves represent the min distances. (it takes the min distance to each endpoint and averages them--graph with clones it takes whichever clone endpoints have the smallest average)
    let orig_min_dist = min_dist_as_edges(&road_graph, &orig_min_dist_to_nodes);
    let modified_min_dist = min_dist_as_edges(&no_left_turns_graph, &modified_min_dist_to_nodes);

    // subtract difference between original and modified graphs
    let difference_graph = edge_difference(&modified_min_dist, &orig_min_dist);

    // convert into Lines for easy nannou drawing
    let road_lines = make_lines_for_nannou(&difference_graph);

    Model {
        _window,
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
Nannou runs this many times per second to create each frame.
*/
fn view(app: &App, model: &Model, frame: Frame) -> Frame {
    let _win = app.window_rect();
    // Prepare to draw.
    let draw = app.draw();

    for road_line in model.road_lines.iter() {
        draw.line()
            .points(road_line.start, road_line.end)
            .thickness(road_line.thickness) // at some point draw according to geographical size ?
            .hsva(road_line.hue, road_line.saturation, 1.0, road_line.alpha);
    }

    draw.background().hsv(0.73, 0.55, 0.06);
    // Write to the window frame.
    draw.to_frame(app, &frame).unwrap();
    // Return the drawn frame.
    frame
}

fn is_in_bounds(node: &Node) -> bool {
    (node.lon() < MAX_LON)
        & (node.lon() > MIN_LON)
        & (node.lat() < MAX_LAT)
        & (node.lat() > MIN_LAT)
}

fn is_in_outer_bounds(node: &Node) -> bool {
    let outer_max_lon = MAX_LON + LON_RANGE * 1.0;
    let outer_min_lon = MIN_LON - LON_RANGE * 1.0;
    let outer_max_lat = MAX_LAT + LAT_RANGE * 1.0;
    let outer_min_lat = MIN_LAT - LAT_RANGE * 1.0;

    (node.lon() < outer_max_lon)
        & (node.lon() > outer_min_lon)
        & (node.lat() < outer_max_lat)
        & (node.lat() > outer_min_lat)
}

fn dist(pt1: Point2, pt2: Point2) -> f32 {
    ((pt1.x - pt2.x).powi(2) + (pt1.y - pt2.y).powi(2)).sqrt()
}

fn convert_coord(node: &Node) -> Point2 {
    let x = map_range(node.lon(), MIN_LON, MAX_LON, -WIN_W * 0.5, WIN_W * 0.5);
    let y = map_range(node.lat(), MIN_LAT, MAX_LAT, -WIN_W * 0.5, WIN_H * 0.5);
    pt2(x, y)
}

/**
This takes the complicated graph (which might have multiple clones for a given location and multiple edges representing a single road segmetn) and simplifies it to an undirected graph with only one edge between nodes and each node representing a distinct geographical location (i.e. a unique OSM NodeId). The edges represent the minimum distance to get to halfway down that edge. Djikstra's algorithm finds the min distance to nodes, however when we draw the roads using nannou, we draw the lines in between nodes, so this function averages the min distance to each endpoint nodes and makes that average the weight of the edge. Because when we draw the final graph, we don't care about directionality anymore, and simply what is the distance to a given edge, so the return graph is undirected.

Note that the turn-limited graph has multiple "clone" nodes for each location. Because more than one node represents a single individual location, for the weight of the final edge in the return graph, this function chooses whichever edge has endpoints with the miniumum min-distance. Because these various clones represent different possible ways of reaching their location, any of them is a valid point to reach, and so the return graph from this function should accurately show the MINIMUM distance to that road segment.
*/
fn min_dist_as_edges(
    g: &Graph<Node, f32, Directed>,
    min_dist: &HashMap<NodeIndex, Option<f32>, RandomState>,
) -> Graph<Node, f32, Undirected> {
    let mut ret_graph = Graph::<Node, f32, Undirected>::new_undirected();
    // group the nodes by their geographical location (at intersections multiple clones correspond to only one point in the map)
    let unique_locations = group_by_orig_location(g);
    let mut g_to_ret_graph: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    for (location, clones) in unique_locations {
        let new_node_ix = ret_graph.add_node(location.clone());
        for clone in clones {
            g_to_ret_graph.insert(clone, new_node_ix);
        }
    }

    for edge in g.edge_indices() {
        // get the endpoints in g, then find their equivalents in ret_graph
        let (start_g, end_g) = g.edge_endpoints(edge).unwrap();
        let start_ret = *g_to_ret_graph.get(&start_g).unwrap();
        let end_ret = *g_to_ret_graph.get(&end_g).unwrap();
        // check to see if an edge already exists. If not, make a new one with maximum f32 value
        let mut ret_graph_edge = ret_graph.find_edge(start_ret, end_ret);
        if ret_graph_edge.is_none() {
            ret_graph_edge = Some(ret_graph.add_edge(start_ret, end_ret, std::f32::MAX));
        }
        let ret_graph_edge = ret_graph_edge.unwrap();

        // compare the edge weight in g with the edge weight in ret_graph, then use the new edge weight if it's lower.
        let min_dist_to_start = min_dist.get(&start_g).unwrap(); // minimum distance to start node
        let min_dist_to_end = min_dist.get(&end_g).unwrap();
        // if the graph is disconnected, then some nodes might have None value, meaning they're unreachable (so ignore them)
        if min_dist_to_start.is_some() && min_dist_to_end.is_some() {
            let mut weight = (min_dist_to_start.unwrap() + min_dist_to_end.unwrap()) / 2.0;
            let mut cur_weight: &mut f32 = ret_graph.edge_weight_mut(ret_graph_edge).unwrap();
            *cur_weight = if OrderedFloat(*cur_weight)
                .cmp(&OrderedFloat(weight))
                .eq(&Ordering::Less)
            {
                *cur_weight
            } else {
                weight
            };
        }
    }
    ret_graph
}

/**
Groups the nodes in the graph by the value they contain. It creates a hashmap from that value (i.e. the "node weight") to the nodes that contain that value.

This has the effect, of in our example where each node contains an OSM node id representing a geographical coordinate, groups nodes by the geographical location they represent (since in our left turn limited graph, there are clones representing the same intersection).
*/
fn group_by_orig_location(
    g: &Graph<Node, f32, Directed>,
) -> HashMap<&Node, Vec<NodeIndex>, RandomState> {
    let mut unique_locations: HashMap<&Node, Vec<NodeIndex>, RandomState> = HashMap::new();
    for node_ix in g.node_indices() {
        let osm_node = g.node_weight(node_ix).unwrap();
        let node_list = unique_locations.entry(osm_node).or_insert(Vec::new());
        node_list.push(node_ix);
    }
    unique_locations
}

/**
This takes two IDENTICALLY STRUCTURED graphs and returns another identically structured graph with the difference between their edge weights (for each edge, the g1 weight minus the g2 weight).

This can show the difference between two pathfinding algorithms for the same road graph (e.g. where one limited certain kinds of turns). The two graphs MUST be identical, except for their edge weights. More specifically, the number of nodes and edges must be the same, the way those edges/nodes are connected must be the same, and the weights of all nodes must be the same. If this isn't true, this function will probably panic or do something weird.
*/
fn edge_difference(
    g1: &Graph<Node, f32, Undirected>,
    g2: &Graph<Node, f32, Undirected>,
) -> Graph<Node, f32, Undirected> {
    // build a hashmap associating the OpenStreetMap node id with node indices from the petgraph graph (the OSM id is the actual values stored in the petgraph graph nodes)
    let g1_nodes: HashMap<i64, NodeIndex> = g1
        .node_indices()
        .map(|node_ix| {
            let osm_node_id = g1.node_weight(node_ix).unwrap().id.0;
            (osm_node_id, node_ix) // note here the node_ix refers to the index of the node in the petgraph Graph, while the osm_node_id refers to the OpenStreetMap node id
        })
        .collect();

    // using the fact that in both graphs equivalent nodes will share the same value (OSM id), we can now build a hashmap associating nodes in g1 with their equivalent nodes in g2 by finding which graph nodes contain the same OSM node id
    let g1_to_g2: HashMap<NodeIndex, NodeIndex> = g2
        .node_indices()
        .map(|g2_node_ix| {
            let osm_node_id = g2.node_weight(g2_node_ix).unwrap().id.0;
            let g1_node_ix = *g1_nodes.get(&osm_node_id).unwrap();
            (g1_node_ix, g2_node_ix)
        })
        .collect();

    // now we build our new graph's nodes, making the clones of the ids in g1
    // we also at the same time build a hashmap associating the g1 nodes with these new graph nodes
    let mut ret_graph = Graph::<Node, f32, Undirected>::new_undirected();
    let mut g1_to_ret = HashMap::new();
    for g1_node_ix in g1.node_indices() {
        let new_node_ix = ret_graph.add_node(g1.node_weight(g1_node_ix).unwrap().clone());
        g1_to_ret.insert(g1_node_ix, new_node_ix);
    }

    // now we use the various hashmaps we previously created associating nodes between the three different graphs to help us rebuild the edges
    for edge_ix in g1.edge_indices() {
        // get g1 endpoints
        let (g1_start, g1_end) = g1.edge_endpoints(edge_ix).unwrap();
        // get g2 equivalents
        let g2_start = *g1_to_g2.get(&g1_start).unwrap();
        let g2_end = *g1_to_g2.get(&g1_end).unwrap();
        // get ret_graph equivalents
        let ret_start = *g1_to_ret.get(&g1_start).unwrap();
        let ret_end = *g1_to_ret.get(&g1_end).unwrap();
        //get g1 edge weight
        let g1_weight = *g1.edge_weight(edge_ix).unwrap();
        // get g2 edge weight
        let g2_weight = *g2
            .edge_weight(g2.find_edge(g2_start, g2_end).unwrap())
            .unwrap();
        // make new edge with weight being difference between the edge weights of the g1 and g2 graphs
        ret_graph.add_edge(ret_start, ret_end, g1_weight - g2_weight);
    }
    ret_graph
}

fn make_lines_for_nannou(g: &Graph<Node, f32, Undirected>) -> Vec<Line> {
    let test_vec: Vec<()> = g
        .edge_indices()
        .map(|e_ix| println!("edge_weight: {}", g.edge_weight(e_ix).unwrap()))
        .collect();
    Vec::new()
}

fn color_roads(road_graph: &Graph<Node, f32, Directed>, test_node: NodeIndex) -> Vec<Line> {
    let node_indices: Vec<NodeIndex> = road_graph.node_indices().collect();
    let num_nodes = node_indices.len();
    let mut rng = rand::thread_rng();
    let rand1 = rng.gen_range(0, num_nodes);
    let start: NodeIndex = road_graph.node_indices().nth(rand1).unwrap();
    let start = test_node; // TESTING

    let min_dist = djikstra_float(&road_graph, start);
    let max_weight = min_dist
        .values()
        .filter(|dist| dist.is_some())
        .map(|float_dist| (float_dist.unwrap() * 10000.) as u32)
        .max()
        .unwrap();

    let num_hue_cycles = 2; // number of times the hue range cycles through before reaching the farthest point.
    let sat_cycles = 10; // number of times saturation cycles through before reaching the farthest point
    let mut road_lines: Vec<Line> = Vec::new();
    for edge in road_graph.raw_edges() {
        let source = road_graph
            .node_weight(edge.source())
            .map(|node| convert_coord(node));
        let target = road_graph
            .node_weight(edge.target())
            .map(|node| convert_coord(node));

        // modify min_dist so it only stores one of the 4 clone nodes
        if source.is_some() && target.is_some() {
            // find weight (which is the f32 path distance from the "start" node to the source node of this edge)
            let weight = min_dist.get(&edge.target()).unwrap();
            // MAKES COLORS ROTATE CYCLICALLY
            let weight = (weight.unwrap_or(0.0) * 10000.0) as u32; // make an integer so you can use % operator
            let weight_hue = weight % (max_weight / num_hue_cycles);
            // TRANSFORM WEIGHT INTO HUE (0.0-1.0) / THICKNESS (1.0+) VALUES
            let hue = map_range(weight_hue, 0, max_weight / num_hue_cycles, HUE_MIN, HUE_MAX);

            // MAKE SATURATION ROTATE CYCLICALLY
            let weight = weight % (max_weight / sat_cycles); // make saturation cycle faster than hue
            let saturation = map_range(weight, 0, max_weight / sat_cycles, 0.5, 1.0);
            //let saturation = 1.0; // if you want no change/cycling of saturation

            // if you want thickness to vary by path distance from start:
            //let thickness = map_range(weight, 0, max_weight, 3.5, 1.0);
            let thickness = 1.0;
            let alpha = 1.0;

            road_lines.push(Line {
                start: source.unwrap(),
                end: target.unwrap(),
                thickness,
                hue,
                saturation,
                alpha,
            });
        }
    }
    road_lines
}

// compactly stores the data to use for the nannou's draw.line() builder, so no need to recompute things when rendering
struct Line {
    start: Point2<f32>,
    end: Point2<f32>,
    thickness: f32,
    hue: f32,
    saturation: f32,
    alpha: f32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct DistFloat {
    dist: OrderedFloat<f32>,
    node: NodeIndex,
}

impl Ord for DistFloat {
    fn cmp(&self, other: &DistFloat) -> Ordering {
        other
            .dist
            .cmp(&self.dist)
            .then_with(|| self.node.index().cmp(&other.node.index()))
    }
}

impl PartialOrd for DistFloat {
    fn partial_cmp(&self, other: &DistFloat) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn djikstra_float(
    g: &Graph<Node, f32, Directed>,
    start: NodeIndex<u32>,
) -> HashMap<NodeIndex, Option<f32>, RandomState> {
    if g.node_weight(start).is_some() {
        let mut min_dist: HashMap<NodeIndex, f32, RandomState>; // final return value; maps node indices to their min distance from the start
        let mut edge_dist: BinaryHeap<DistFloat> = BinaryHeap::new(); // potential edges to explore; each Dist stores a node and a distance leading up to that node (which may not be the min distance)
                                                                      // Djikstra's formula works by always selecting the minimum of those potential edges (hence why it's a BinaryHeap (Priority Queue)

        // initialize min_dist
        min_dist = HashMap::new();
        for node in g.node_indices() {
            min_dist.insert(node, std::f32::INFINITY);
        }
        // set starting point to zero
        min_dist.insert(start, 0.0);
        // if it's a modified graph, multiple clones represent one location (osm_id), so all the nodes for the starting location need to have minimum distance of zero.

        let start_indices: Vec<NodeIndex> = g
            .node_indices()
            .filter(|&node_ix| *g.node_weight(start).unwrap() == *g.node_weight(node_ix).unwrap())
            .collect();

        for start_ix in start_indices {
            min_dist.insert(start_ix, 0.0);
            let edges_from_start = g.edges_directed(start_ix, Outgoing);
            edges_from_start
                .map(|edge| DistFloat {
                    node: edge.target(),
                    dist: OrderedFloat(*edge.weight()),
                })
                .for_each(|edge| edge_dist.push(edge));
        }

        /*
                println!("transformed graph ****");
                for edge in g.edge_indices() {
                    let (start,end) = g.edge_endpoints(edge).unwrap();
                    let start_weight = g.node_weight(start).unwrap();
                    let end_weight = g.node_weight(end).unwrap();
                        println!("({},{} -> {},{})", start_weight.id.0, start.index(), end_weight.id.0, end.index());
                       // println!("({},{})",start_weight.id.0, end_weight.id.0);
                    }
        */
        // initialize edge_dist, the priority queue (binary heap), with all outgoing edges from start
        let edges_from_start = g.edges_directed(start, Outgoing);
        edge_dist = edges_from_start
            .map(|edge| DistFloat {
                node: edge.target(),
                dist: OrderedFloat(*edge.weight()),
            })
            .collect();

        // traverse graph, adding one node at a time (choose the one with the lowest accumulated path distance)
        while !edge_dist.is_empty() {
            let next = edge_dist.pop().unwrap(); // take minimum

            // if it's lower than the currently saved min distance value, then update min_dist and add its edges to the edge priority queue
            if next.dist.into_inner() < *min_dist.get(&next.node).unwrap() {
                min_dist.insert(next.node, next.dist.into_inner());

                let outgoing = g.edges_directed(next.node, Outgoing);
                let outgoing: Vec<DistFloat> = outgoing
                    .map(|edge| DistFloat {
                        node: edge.target(),
                        dist: OrderedFloat(*edge.weight() + next.dist.into_inner()),
                    })
                    .collect();
                for dist in outgoing {
                    edge_dist.push(dist);
                }
            }
        }

        let min_dist: HashMap<NodeIndex, Option<f32>, RandomState> = min_dist
            .into_iter()
            .map(|(node, dist)| {
                if dist == std::f32::INFINITY {
                    (node, None)
                } else {
                    (node, Some(dist))
                }
            })
            .collect();

        min_dist
    } else {
        HashMap::new()
    }
}

/**
Takes pairings of neighbors/internal nodes and returns them sorted such that the neighbors are arranged clockwise around the given center (starting with whichever is in the northwest quadrant closest to west)
*/
fn get_clockwise_order2(
    g: &Graph<Node, f32, Directed>,
    center: NodeIndex,
    neighbors: Vec<(NodeIndex, NodeIndex)>,
) -> Vec<(NodeIndex, NodeIndex)> {
    let center = g.node_weight(center).unwrap();
    let mut neighbor_angles: Vec<(f64, (NodeIndex, NodeIndex))> = neighbors
        .into_iter()
        .map(|(neighbor_node, internal_node)| {
            let actual_map_node = g.node_weight(neighbor_node).unwrap();
            let dx = actual_map_node.lon() - center.lon();
            let dy = actual_map_node.lat() - center.lat();
            let angle = dy.atan2(dx);
            (angle, (neighbor_node, internal_node))
        })
        .collect();

    neighbor_angles.sort_by(|a, b| (-a.0).partial_cmp(&(-b.0)).unwrap());
    neighbor_angles
        .into_iter()
        .map(|(_angle, node_tuple)| node_tuple)
        .collect()
}

fn prohibit_left_turns2(g: &Graph<Node, f32, Directed>) -> Graph<Node, f32, Directed> {

    // if it's a two way, we don't need access to the graph, just choose the opposite one
    // if it's a four way, if you have access to the osm_nodes, then you don't need access to the graph
    // if it's threeway
    // instead of passing the roadnode objects, just pass one big object (or a func) that given an osm node and a node_index, will return the clones to connect to


    // make the new graph
    // simultaneously for each road node create
    //      a RoadNode object, its implementation depending on the neighbor count
    //      clones in the new graph (accessible through the RoadNode object)
    // if not a RoadNode object, then pass a tuple of
    let mut new_g: Graph<Node, f32, Directed> = Graph::new();
    let orig_to_clones = add_clones(&g,&mut new_g);
    add_edges(&g, &mut new_g, orig_to_clones);
    //let clones_map: HashMap<NodeIndex,RoadNode,RandomState> = make_clones(g, &mut new_g);
    //add_edges(g, &mut new_g, clones_map);

    // to construct itself it needs access to the coordineate of all the points which means giving it either the nodes themselvs or the node_indices plus the graph\\//
    // it also needs to store the node indices of the clones
    // it also needs to store the osm ids of the neighbors mapped to those clones
    // however, no edges need to be made yet, which clone goes with which neighbor is randomly assigned on construction and the clone list is then sorted
    // args: (vec<neighbor_nodes>, vec<clone_indices>, center_node)
    // get_clones args: (osm_id) -> Vec<clone_indices>
    // need a

    // add_edges:
    //      for each original node in graph, look up its RoadNode (cur_rn)
    //      for each neighbor
    //          look up original edge weight
    //          look up the RoadNode of that neighbor (neighbor_rn)
    //          look up the clone that points in that direction (osm_id)
    //          using neighbor_rn, enter the osm id for this node and get back clones to connect to (Vec<NodeIndex>)
    //          add edges for each (nodeindex,penalty) you get back
    new_g
}

fn add_edges(g: &Graph<Node, f32, Directed>, new_g: &mut Graph<Node, f32, Directed>, orig_to_clones: HashMap<NodeIndex, RoadNode,RandomState>) {
    for (node_ix, road_node) in orig_to_clones {
        // fetch the neighbors of this clone
        for neighbor in g.neighbors(node_ix) {
            let neighbor_osm_node = g.node_weight(neighbor).unwrap().clone();
            let clones = road_node.get_clones(neighbor_osm_node);
        }
    }
    /*
    let edges_to_add = orig_to_clones.iter()
        .map(|(&node_ix,&road_node)| {
            g.neighbors(node_ix)
                .map(|neighbor|{
                    let neighbor_osm_node = *g.node_weight(neighbor).unwrap();
                    road_node.get_clones(neighbor_osm_node).into_iter()
                        .flat_map(|(clone,penalty)|{
                            let neighbor_road_clone = orig_to_clones.get(&neighbor).unwrap();
                            let neighbor_clone = neighbor_road_clone.get_u_turn(g.node_weight(node_ix).unwrap());
                            // TODO: finish this part
                        }
                        )
                })
                .map(|neighbor_osm_node| )
        }
        )*/
}

fn add_clones(g: &Graph<Node, f32, Directed>, new_g: &mut Graph<Node, f32, Directed>) -> HashMap<NodeIndex, RoadNode> {
    // for each node in g create clones, bundle into a RoadNode, then return the hash from original graph nodes to the RoadNode
    HashMap::new()
}

// instead of the RoadNode we do this:
// hashmap of NodeIndex (orig) to Vec<(NodeIndex,NodeIndex)>
//

fn sort_clockwise(
    center: &Node,
    neighbors_and_clones: &Vec<(Node, NodeIndex)>,
) -> Vec<(Node, NodeIndex)> {
    Vec::new()
}



/**
I wanted to use trait objects here for dynamic dispatch, (a trait for RoadNode and structs like "3Way" or "FourWay" or "t_intersection") but I couldn't figure out how to make it work in Rust.
*/
struct RoadNode {
    center: Node,
    neighbor_clone_pairs: Vec<(Node, NodeIndex)>
}

impl RoadNode {
    fn new(center: Node, neighbors: Vec<Node>, clones: Vec<NodeIndex>) -> RoadNode{
        assert_eq!(neighbors.len(),clones.len());
            let mut neighbor_clone_pairs: Vec<(Node, NodeIndex)> = neighbors.into_iter()
                .zip(clones.into_iter())
                .collect();
            neighbor_clone_pairs = sort_clockwise(&center,&neighbor_clone_pairs);

        RoadNode{center, neighbor_clone_pairs}
    }

    fn get_clones(&self, neighbor: Node) -> Vec<(NodeIndex, f32)> {
        let mut ret = Vec::new();
        let num_neighbors = self.neighbor_clone_pairs.len();
        if num_neighbors == 2 {
            let uturn_clone = self.get_x_rotations_from(&neighbor,0);
            let straight_ahead_clone = self.get_x_rotations_from(&neighbor,1);
             vec!((uturn_clone, UTURN_PENALTY),(straight_ahead_clone, STRAIGHT_PENALTY));
        } else if num_neighbors == 4 {

        } else {

        }
        ret
    }
    /**
    Assuming the "neighbor" argument is in fact one of the neighbors of this intersection, this provides the clone pointing in a new direction (the direction x clockwise rotations away from this neighbor).
    For example, in a 4 way intersection, if neighbor represents the neighbor to the west of the intersection, and you let x=3, then what will be returned is the clone pointing to the south neighbor (3 rotations away).

    If the neighbor argument isn't actually a neighbor of this intersection, then this will panic.
    */
    fn get_x_rotations_from(&self, neighbor: &Node, x: usize) -> NodeIndex {
        let num_clones = self.neighbor_clone_pairs.len();
        let neighbor_index = self.neighbor_clone_pairs.iter().position(|(neighbor_node,clone)| neighbor_node.eq(neighbor)).unwrap();
        self.neighbor_clone_pairs.get((neighbor_index + x) % num_clones).unwrap().1
    }

    fn get_u_turn(&self, neighbor: &Node) -> NodeIndex {
        self.get_x_rotations_from(neighbor,0)
    }
}


/**
TODO: put in separate file
*/
mod tests {
    use super::*;

    /*
    // TESTS FOR CONSTRUCTING TURN LIMITED GRAPH
    #[test]
    fn test_empty_graph_construction() {
        let test_graph = Graph::<Node, f32, Directed>::new();
        let result = prohibit_left_turns(&test_graph);
        assert_eq!(result.node_count(), 0);
        assert_eq!(result.edge_count(), 0);
    }

    #[test]
    fn test_one_node_construction() {
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let n0 = test_graph.add_node(node0);
        let result = prohibit_left_turns(&test_graph);
        assert_eq!(result.node_count(), 1);
        assert_eq!(result.edge_count(), 0);
    }

    #[test]
    fn test_two_node_construction() {
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let n0 = test_graph.add_node(node0);
        let n1 = test_graph.add_node(node1);
        test_graph.add_edge(n0, n1, 5.0);
        let result = prohibit_left_turns(&test_graph);
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 1);
        assert_eq!(
            5.0,
            *result
                .edge_weight(result.edge_indices().next().unwrap())
                .unwrap()
        );
    }

    #[test]
    fn test_two_intersection_construction() {
        // this test tests when two 4 way intersections are adjacent to each other
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: 1,
        };
        let node2: Node = Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 1,
            decimicro_lon: 10,
        };
        let node3: Node = Node {
            id: NodeId(3),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: -1,
        };
        let node4: Node = Node {
            id: NodeId(4),
            tags: Default::default(),
            decimicro_lat: -1,
            decimicro_lon: -10,
        };
        let node5: Node = Node {
            id: NodeId(5),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: 10,
        };
        let node6: Node = Node {
            id: NodeId(6),
            tags: Default::default(),
            decimicro_lat: 20,
            decimicro_lon: -1,
        };
        let node7: Node = Node {
            id: NodeId(7),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: -10,
        };
        let mut test_graph = Graph::<Node, f32, Directed>::new();

        let n0 = test_graph.add_node(node0);
        let n1 = test_graph.add_node(node1);
        let n2 = test_graph.add_node(node2);
        let n3 = test_graph.add_node(node3);
        let n4 = test_graph.add_node(node4);
        let n5 = test_graph.add_node(node5);
        let n6 = test_graph.add_node(node6);
        let n7 = test_graph.add_node(node7);
        test_graph.add_edge(n0, n1, 1.0); // the edge weights don't actually matter here as we're only testing the structure of the graph
        test_graph.add_edge(n1, n0, 1.0);
        test_graph.add_edge(n0, n2, 1.0);
        test_graph.add_edge(n2, n0, 1.0);
        test_graph.add_edge(n0, n3, 1.0);
        test_graph.add_edge(n3, n0, 1.0);
        test_graph.add_edge(n0, n4, 1.0);
        test_graph.add_edge(n4, n0, 1.0);
        test_graph.add_edge(n3, n5, 1.0);
        test_graph.add_edge(n5, n3, 1.0);
        test_graph.add_edge(n3, n6, 1.0);
        test_graph.add_edge(n6, n3, 1.0);
        test_graph.add_edge(n3, n7, 1.0);
        test_graph.add_edge(n7, n3, 1.0);

        let result = prohibit_left_turns(&test_graph);
        assert_eq!(14, result.node_count());
        assert_eq!(22, result.edge_count());
    }

    #[test]
    fn test_three_intersection_construction() {
        // this test tests when three 4 way intersections are connected to each other in a line
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: 1,
        };
        let node2: Node = Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 1,
            decimicro_lon: 10,
        };
        let node3: Node = Node {
            id: NodeId(3),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: -1,
        };
        let node4: Node = Node {
            id: NodeId(4),
            tags: Default::default(),
            decimicro_lat: -1,
            decimicro_lon: -10,
        };
        let node5: Node = Node {
            id: NodeId(5),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: 10,
        };
        let node6: Node = Node {
            id: NodeId(6),
            tags: Default::default(),
            decimicro_lat: 20,
            decimicro_lon: -1,
        };
        let node7: Node = Node {
            id: NodeId(7),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: -10,
        };
        let node8: Node = Node {
            id: NodeId(8),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: -10,
        };
        let node9: Node = Node {
            id: NodeId(9),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: -20,
        };
        let node10: Node = Node {
            id: NodeId(10),
            tags: Default::default(),
            decimicro_lat: 20,
            decimicro_lon: -10,
        };
        let mut test_graph = Graph::<Node, f32, Directed>::new();

        let n0 = test_graph.add_node(node0);
        let n1 = test_graph.add_node(node1);
        let n2 = test_graph.add_node(node2);
        let n3 = test_graph.add_node(node3);
        let n4 = test_graph.add_node(node4);
        let n5 = test_graph.add_node(node5);
        let n6 = test_graph.add_node(node6);
        let n7 = test_graph.add_node(node7);
        let n8 = test_graph.add_node(node8);
        let n9 = test_graph.add_node(node9);
        let n10 = test_graph.add_node(node10);
        test_graph.add_edge(n0, n1, 1.0); // the edge weights don't actually matter here as we're only testing the structure of the graph
        test_graph.add_edge(n1, n0, 1.0);
        test_graph.add_edge(n0, n2, 1.0);
        test_graph.add_edge(n2, n0, 1.0);
        test_graph.add_edge(n0, n3, 1.0);
        test_graph.add_edge(n3, n0, 1.0);
        test_graph.add_edge(n0, n4, 1.0);
        test_graph.add_edge(n4, n0, 1.0);
        test_graph.add_edge(n3, n5, 1.0);
        test_graph.add_edge(n5, n3, 1.0);
        test_graph.add_edge(n3, n6, 1.0);
        test_graph.add_edge(n6, n3, 1.0);
        test_graph.add_edge(n3, n7, 1.0);
        test_graph.add_edge(n7, n3, 1.0);
        test_graph.add_edge(n7, n8, 1.0);
        test_graph.add_edge(n8, n7, 1.0);
        test_graph.add_edge(n7, n9, 1.0);
        test_graph.add_edge(n9, n7, 1.0);
        test_graph.add_edge(n7, n10, 1.0);
        test_graph.add_edge(n10, n7, 1.0);

        let result = prohibit_left_turns(&test_graph);
        assert_eq!(20, result.node_count());
        assert_eq!(32, result.edge_count());
    }

    #[test]
    fn test_three_intersection_construction2() {
        // this test tests when three 4-way intersections are all connected to each other
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: 1,
        };
        let node2: Node = Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 1,
            decimicro_lon: 10,
        };
        let node3: Node = Node {
            id: NodeId(3),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: -1,
        };
        let node5: Node = Node {
            id: NodeId(5),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: 10,
        };
        let node6: Node = Node {
            id: NodeId(6),
            tags: Default::default(),
            decimicro_lat: 20,
            decimicro_lon: -1,
        };
        let node7: Node = Node {
            id: NodeId(7),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: -10,
        };
        let node9: Node = Node {
            id: NodeId(9),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: -20,
        };
        let node10: Node = Node {
            id: NodeId(10),
            tags: Default::default(),
            decimicro_lat: 20,
            decimicro_lon: -10,
        };
        let mut test_graph = Graph::<Node, f32, Directed>::new();

        let n0 = test_graph.add_node(node0);
        let n1 = test_graph.add_node(node1);
        let n2 = test_graph.add_node(node2);
        let n3 = test_graph.add_node(node3);
        let n5 = test_graph.add_node(node5);
        let n6 = test_graph.add_node(node6);
        let n7 = test_graph.add_node(node7);
        let n9 = test_graph.add_node(node9);
        let n10 = test_graph.add_node(node10);
        test_graph.add_edge(n0, n1, 1.0); // the edge weights don't actually matter here as we're only testing the structure of the graph
        test_graph.add_edge(n1, n0, 1.0);
        test_graph.add_edge(n0, n2, 1.0);
        test_graph.add_edge(n2, n0, 1.0);
        test_graph.add_edge(n0, n3, 1.0);
        test_graph.add_edge(n3, n0, 1.0);

        test_graph.add_edge(n3, n5, 1.0);
        test_graph.add_edge(n5, n3, 1.0);
        test_graph.add_edge(n3, n6, 1.0);
        test_graph.add_edge(n6, n3, 1.0);
        test_graph.add_edge(n3, n7, 1.0);
        test_graph.add_edge(n7, n3, 1.0);

        test_graph.add_edge(n7, n0, 1.0);
        test_graph.add_edge(n0, n7, 1.0);
        test_graph.add_edge(n7, n9, 1.0);
        test_graph.add_edge(n9, n7, 1.0);
        test_graph.add_edge(n7, n10, 1.0);
        test_graph.add_edge(n10, n7, 1.0);

        let result = prohibit_left_turns(&test_graph);
        assert_eq!(18, result.node_count());
        assert_eq!(30, result.edge_count());
    }

    // TESTS FOR "DECONSTRUCTING" THE GRAPH (transforming it into an undirected graph where there is only one edge per road segment and only one node per OSM NodeId and edge weights are the average of the min distance to the endpoints)
    #[test]
    fn test_no_nodes_deconstruction() {
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let mut min_dist = HashMap::new();
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 0);
    }

    #[test]
    fn test_one_node_deconstruction() {
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let mut min_dist = HashMap::new();
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let n0 = test_graph.add_node(node0);
        min_dist.insert(n0, Some(0.0)); // one node is "reachable"
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 1);
        min_dist.clear();
        min_dist.insert(n0, None); // the node is not "reachable" (doesn't really make sense but worth testing)
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 1);
    }

    #[test]
    fn test_two_nodes_disconnected_deconstruction() {
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let mut min_dist = HashMap::new();
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let n0 = test_graph.add_node(node0);
        let node0: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let n1 = test_graph.add_node(node0);
        min_dist.insert(n0, None); // one node is "reachable"
        min_dist.insert(n1, None); // one node is "reachable"
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 0);
        min_dist.clear();
        min_dist.insert(n0, Some(10.0)); // one node is "reachable"
        min_dist.insert(n1, Some(5.0)); // one node is "reachable"
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 0);
    }

    #[test]
    fn test_two_nodes_one_edge_deconstruction() {
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let mut min_dist = HashMap::new();
        let n0 = test_graph.add_node(node0);
        let n1 = test_graph.add_node(node1);
        test_graph.add_edge(n0, n1, 10.0);
        min_dist.insert(n0, Some(0.0));
        min_dist.insert(n1, Some(10.0));
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 1);
        assert_eq!(
            *result
                .edge_weight(result.edge_indices().next().unwrap())
                .unwrap(),
            5.0
        );
    }

    #[test]
    fn test_two_nodes_two_edges_deconstruction() {
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let mut min_dist = HashMap::new();
        let n0 = test_graph.add_node(node0);
        let n1 = test_graph.add_node(node1);
        test_graph.add_edge(n0, n1, 100.0);
        test_graph.add_edge(n1, n0, 200.0);
        min_dist.insert(n0, Some(10.0));
        min_dist.insert(n1, Some(20.0));
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 1);
        assert_eq!(
            *result
                .edge_weight(result.edge_indices().next().unwrap())
                .unwrap(),
            15.0
        );
    }

    #[test]
    fn test_two_clones_no_edges_deconstruction() {
        // because there are two "clones" (i.e. nodes in the graph representing the same OSM NodeId), the final graph should have only one node and no edges
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let mut min_dist = HashMap::new();
        let n0 = test_graph.add_node(node0);
        let n1 = test_graph.add_node(node1);
        min_dist.insert(n0, Some(10.0));
        min_dist.insert(n1, Some(20.0));
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 1);
        assert_eq!(result.edge_count(), 0);
    }

    #[test]
    fn test_multi_clones_multi_edges_deconstruction() {
        let node0_clone0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node0_clone1: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1_clone0: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1_clone1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1_clone2: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node3: Node = Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let mut test_graph = Graph::<Node, f32, Directed>::new();
        let mut min_dist = HashMap::new();
        let n0_c0 = test_graph.add_node(node0_clone0);
        let n0_c1 = test_graph.add_node(node0_clone1);
        let n1_c0 = test_graph.add_node(node1_clone0);
        let n1_c1 = test_graph.add_node(node1_clone1);
        let n1_c2 = test_graph.add_node(node1_clone2);
        let n3 = test_graph.add_node(node3);
        min_dist.insert(n0_c0, Some(0.0));
        min_dist.insert(n0_c1, Some(10.0));
        min_dist.insert(n1_c0, Some(20.0));
        min_dist.insert(n1_c1, Some(30.0));
        min_dist.insert(n1_c2, None);
        min_dist.insert(n3, None);
        test_graph.add_edge(n0_c0, n1_c0, 7.0); // these edge weights don't actually matter
        test_graph.add_edge(n0_c1, n1_c0, 7.0);
        test_graph.add_edge(n1_c0, n0_c0, 7.0);
        test_graph.add_edge(n1_c1, n0_c1, 7.0);
        test_graph.add_edge(n1_c0, n0_c1, 7.0);
        test_graph.add_edge(n1_c2, n0_c0, 7.0);
        test_graph.add_edge(n3, n1_c0, 7.0);
        let result = min_dist_as_edges(&test_graph, &min_dist);
        assert_eq!(result.node_count(), 3);
        assert_eq!(result.edge_count(), 2);
        assert_eq!(
            *result
                .edge_weight(result.edge_indices().next().unwrap())
                .unwrap(),
            10.0
        );
    }

    // TEST DIFFERENCE GRAPH
    #[test]
    fn test_empty_graphs_diff() {
        let mut g1 = Graph::<Node, f32, Undirected>::new_undirected();
        let mut g2 = Graph::<Node, f32, Undirected>::new_undirected();
        let result = edge_difference(&g1, &g2);
        assert_eq!(result.node_count(), 0);
        assert_eq!(result.edge_count(), 0);
    }

    #[test]
    fn test_no_edge_graphs_diff() {
        let mut g1 = Graph::<Node, f32, Undirected>::new_undirected();
        let mut g2 = Graph::<Node, f32, Undirected>::new_undirected();
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        g1.add_node(node0.clone());
        g1.add_node(node1.clone());
        g2.add_node(node0);
        g2.add_node(node1);
        let result = edge_difference(&g1, &g2);
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 0);
    }

    #[test]
    fn test_one_edge_graphs_diff() {
        let mut g1 = Graph::<Node, f32, Undirected>::new_undirected();
        let mut g2 = Graph::<Node, f32, Undirected>::new_undirected();
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let g1n1 = g1.add_node(node0.clone());
        let g1n2 = g1.add_node(node1.clone());
        g1.add_edge(g1n1, g1n2, 5.0);
        let g2n1 = g2.add_node(node0);
        let g2n2 = g2.add_node(node1);
        g2.add_edge(g2n1, g2n2, 2.0);
        let result = edge_difference(&g1, &g2);
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 1);
        assert_eq!(
            *result
                .edge_weight(result.edge_indices().next().unwrap())
                .unwrap(),
            3.0
        );
    }

    #[test]
    fn test_multi_edge_disconnect_diff() {
        // tests case where there are multiple edges as well as disconnected components in the graph
        let mut g1 = Graph::<Node, f32, Undirected>::new_undirected();
        let mut g2 = Graph::<Node, f32, Undirected>::new_undirected();
        let node0: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node2: Node = Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node3: Node = Node {
            id: NodeId(3),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let g1n0 = g1.add_node(node0.clone());
        let g1n1 = g1.add_node(node1.clone());
        let g1n2 = g1.add_node(node2.clone());
        let g1n3 = g1.add_node(node3.clone());
        g1.add_edge(g1n0, g1n1, 5.0);
        g1.add_edge(g1n1, g1n2, 15.0);
        g1.add_edge(g1n2, g1n0, 25.0);
        let g2n0 = g2.add_node(node0);
        let g2n1 = g2.add_node(node1);
        let g2n2 = g2.add_node(node2);
        let g2n3 = g2.add_node(node3);
        g2.add_edge(g2n0, g2n1, 3.0);
        g2.add_edge(g2n1, g2n2, 7.0);
        g2.add_edge(g2n2, g2n0, 21.0);
        let result = edge_difference(&g1, &g2);
        assert_eq!(result.node_count(), 4);
        assert_eq!(result.edge_count(), 3);
        assert_eq!(
            *result
                .edge_weight(result.edge_indices().next().unwrap())
                .unwrap(),
            2.0
        );
        let exp_edge_weights: HashSet<OrderedFloat<f32>> =
            [OrderedFloat(2.0), OrderedFloat(8.0), OrderedFloat(4.0)]
                .iter()
                .cloned()
                .collect();
        let edge_weights: HashSet<OrderedFloat<f32>, RandomState> = result
            .edge_indices()
            .map(|e_ix| OrderedFloat(*result.edge_weight(e_ix).unwrap()))
            .collect();
        assert_eq!(exp_edge_weights, edge_weights);
    } */

    // TESTS FOR FOURWAY OBJECT
    //#[test]
    /*
    fn test_construct_fourway() {
        // create graph with 1 intersection
        // construct the 4 way
        // query all directions for each neighbor and verify it works right
        // manually change test so I redistribute the order I add nodes and whatnot just to doublecheck nothing accidntally works
        let mut test_graph: Graph<Node, f32, Directed> = Graph::new();
        let center: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 1,
            decimicro_lon: -10,
        };
        let node2: Node = Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: 1,
        };
        let node3: Node = Node {
            id: NodeId(3),
            tags: Default::default(),
            decimicro_lat: -1,
            decimicro_lon: 10,
        };
        let node4: Node = Node {
            id: NodeId(4),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: -1,
        };

        let node2 = test_graph.add_node(node2);
        let node3 = test_graph.add_node(node3);
        let center = test_graph.add_node(center);
        let node1 = test_graph.add_node(node1);
        let node4 = test_graph.add_node(node4);

        test_graph.add_edge(node3, center, 1.0);
        test_graph.add_edge(center, node3, 1.0);
        test_graph.add_edge(center, node1, 1.0);
        test_graph.add_edge(center, node2, 1.0);
        test_graph.add_edge(center, node4, 1.0);
        test_graph.add_edge(node1, center, 1.0);
        test_graph.add_edge(node2, center, 1.0);
        test_graph.add_edge(node4, center, 1.0);

        let result = prohibit_left_turns(&test_graph);
        /*println!("TEST FOUR WAY ");
        for edge in result.edge_indices() {
            let (start,end) = result.edge_endpoints(edge).unwrap();
            let start_osm = result.node_weight(start).unwrap().id.0;
            let end_osm = result.node_weight(end).unwrap().id.0;
            println!("({},{})->({},{})", start_osm, start.index(), end_osm, end.index());
        }*/
        assert_eq!(8, result.node_count());
    }

    #[test]
    fn test_construct_two_fourways() {
        let mut test_graph: Graph<Node, f32, Directed> = Graph::new();
        let center: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let node1: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 1,
            decimicro_lon: -10,
        };
        let node2: Node = Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: 1,
        };
        let node3: Node = Node {
            id: NodeId(3),
            tags: Default::default(),
            decimicro_lat: -1,
            decimicro_lon: 10,
        };
        let node4: Node = Node {
            id: NodeId(4),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: -1,
        };
        let node5: Node = Node {
            id: NodeId(5),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: -1,
        };
        let node6: Node = Node {
            id: NodeId(6),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: -1,
        };
        let node7: Node = Node {
            id: NodeId(7),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: -1,
        };

        let center = test_graph.add_node(center);
        let node1 = test_graph.add_node(node1);
        let node2 = test_graph.add_node(node2);
        let node3 = test_graph.add_node(node3);
        let node4 = test_graph.add_node(node4);
        let node5 = test_graph.add_node(node5);
        let node6 = test_graph.add_node(node6);
        let node7 = test_graph.add_node(node7);

        test_graph.add_edge(center, node1, 1.0);
        test_graph.add_edge(center, node2, 1.0);
        test_graph.add_edge(center, node3, 1.0);
        test_graph.add_edge(center, node4, 1.0);
        test_graph.add_edge(node1, center, 1.0);
        test_graph.add_edge(node2, center, 1.0);
        test_graph.add_edge(node3, center, 1.0);
        test_graph.add_edge(node4, center, 1.0);

        test_graph.add_edge(node3, node5, 1.0);
        test_graph.add_edge(node3, node6, 1.0);
        test_graph.add_edge(node3, node7, 1.0);
        test_graph.add_edge(node5, node3, 1.0);
        test_graph.add_edge(node6, node3, 1.0);
        test_graph.add_edge(node7, node3, 1.0);

        let result = prohibit_left_turns(&test_graph);
        println!("TEST TWO FOURWAYS");
        for edge in result.edge_indices() {
            let (start, end) = result.edge_endpoints(edge).unwrap();
            let start_osm = result.node_weight(start).unwrap().id.0;
            let end_osm = result.node_weight(end).unwrap().id.0;
            println!(
                "({},{})->({},{})",
                start_osm,
                start.index(),
                end_osm,
                end.index()
            );
        }
        assert_eq!(14, result.node_count());
        assert_eq!(22, result.edge_count());
    }

    // TESTS THE FOURWAY STRUCT TO MAKE SURE ITS CONSTRUCTED CORRECTLY REGARDLESS OF ORDER OF INPUT
    #[test]
    fn test_fourway() {
        let mut test_graph: Graph<Node, f32, Directed> = Graph::new();
        let center: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let west: Node = Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 1,
            decimicro_lon: -10,
        };
        let north: Node = Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 10,
            decimicro_lon: 1,
        };
        let east: Node = Node {
            id: NodeId(3),
            tags: Default::default(),
            decimicro_lat: -1,
            decimicro_lon: 10,
        };
        let south: Node = Node {
            id: NodeId(4),
            tags: Default::default(),
            decimicro_lat: -10,
            decimicro_lon: -1,
        };

        let cent = test_graph.add_node(center.clone());
        let west_ix = test_graph.add_node(west);
        let north_ix = test_graph.add_node(north);
        let east_ix = test_graph.add_node(east);
        let south_ix = test_graph.add_node(south);
        let clone_to_west = test_graph.add_node(center.clone());
        let clone_to_north = test_graph.add_node(center.clone());
        let clone_to_east = test_graph.add_node(center.clone());
        let clone_to_south = test_graph.add_node(center.clone());

        let mut edges = Vec::new();
        edges.push(test_graph.add_edge(clone_to_north, north_ix, 1.0));
        edges.push(test_graph.add_edge(clone_to_east, east_ix, 1.0));
        edges.push(test_graph.add_edge(clone_to_west, west_ix, 1.0));
        edges.push(test_graph.add_edge(clone_to_south, south_ix, 1.0));

        let result = FourWay::new(&test_graph, cent, edges);
        // TEST GOING STRAIGHT ACROSS
        assert_eq!(clone_to_west, result.get_straight_across(east_ix).unwrap());
        assert_eq!(
            clone_to_north,
            result.get_straight_across(south_ix).unwrap()
        );
        assert_eq!(clone_to_east, result.get_straight_across(west_ix).unwrap());
        assert_eq!(
            clone_to_south,
            result.get_straight_across(north_ix).unwrap()
        );
        // TEST TURNING RIGHT
        assert_eq!(clone_to_west, result.get_right_turn(north_ix).unwrap());
        assert_eq!(clone_to_north, result.get_right_turn(east_ix).unwrap());
        assert_eq!(clone_to_east, result.get_right_turn(south_ix).unwrap());
        assert_eq!(clone_to_south, result.get_right_turn(west_ix).unwrap());
        // TEST TURNING LEFT
        assert_eq!(clone_to_west, result.get_left_turn(south_ix).unwrap());
        assert_eq!(clone_to_north, result.get_left_turn(west_ix).unwrap());
        assert_eq!(clone_to_east, result.get_left_turn(north_ix).unwrap());
        assert_eq!(clone_to_south, result.get_left_turn(east_ix).unwrap());
        // TEST GOING BACK
        assert_eq!(clone_to_west, result.get_back(west_ix).unwrap());
        assert_eq!(clone_to_north, result.get_back(north_ix).unwrap());
        assert_eq!(clone_to_east, result.get_back(east_ix).unwrap());
        assert_eq!(clone_to_south, result.get_back(south_ix).unwrap());
    }
*/
    #[test]
    fn test_sort_clockwise() {
        let center: Node = Node {
            id: NodeId(0),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        };
        let west: (Node,NodeIndex) = (
            Node {
                id: NodeId(1),
                tags: Default::default(),
                decimicro_lat: 1,
                decimicro_lon: -10,
            },
            NodeIndex::new(0),
        );
        let north: (Node,NodeIndex) = (
            Node {
                id: NodeId(2),
                tags: Default::default(),
                decimicro_lat: 10,
                decimicro_lon: 1,
            },
            NodeIndex::new(0),
        );
        let east: (Node,NodeIndex) = (
            Node {
                id: NodeId(3),
                tags: Default::default(),
                decimicro_lat: -1,
                decimicro_lon: 10,
            },
            NodeIndex::new(0),
        );
        let south: (Node,NodeIndex) = (
            Node {
                id: NodeId(4),
                tags: Default::default(),
                decimicro_lat: -10,
                decimicro_lon: -1,
            },
            NodeIndex::new(0),
        );
        let mut neighbors_and_clones = Vec::new();
        neighbors_and_clones.push(south.clone());
        neighbors_and_clones.push(west.clone());
        neighbors_and_clones.push(east.clone());
        neighbors_and_clones.push(north.clone());

        let mut exp_result = Vec::new();
        exp_result.push(west);
        exp_result.push(north);
        exp_result.push(east);
        exp_result.push(south);

        let result = sort_clockwise(&center,&neighbors_and_clones);
        assert_eq!(result, exp_result);
    }

    /**
    This just tests to make sure I'm correctly using petgraph's algorithm to test for graph isomorphism
    */
    #[test]
    fn test_isomorphic_graph() {
        // these functions used for petgraph::algo::is_isomorphic_matching to test node/edge equality
        let node_match = |n1: &Node, n2: &Node| {n1.eq(n2)};
        let edge_match = |e1: &f32, e2: &f32| {e1==e2};

        // test empty graphs
        let mut g1: Graph<Node, f32, Directed> = Graph::new();
        let mut g2: Graph<Node, f32, Directed> = Graph::new();
        assert!(is_isomorphic_matching(&g1, &g2, node_match, edge_match));

        // test simple graphs that are the same
        let n1 = g1.add_node(Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        });
        let n2 = g1.add_node(Node {
            id: NodeId(2),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        });
        g1.add_edge(n1,n2,1.0);
        g2 = g1.clone();
        assert!(is_isomorphic_matching(&g1, &g2, node_match, edge_match));

        // test simple graphs that differ by edge weight (but node weights are the same)
        g2.clear_edges();
        g2.add_edge(n1.clone(), n2.clone(), 2.0); // now edge has 2.0 weight not 1.0
        assert!(! is_isomorphic_matching(&g1, &g2, node_match, edge_match));

        // test simple graphs that differ by node weight (but edge weights are the same)
        g2.clear();
        let n1 = g2.add_node(Node {
            id: NodeId(1),
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        });
        let n3 = g2.add_node(Node {
            id: NodeId(3), // this is different from what's in g1
            tags: Default::default(),
            decimicro_lat: 0,
            decimicro_lon: 0,
        });
        g2.add_edge(n1,n3,1.0); // edge weight the same but the nodes are different
        assert!(! is_isomorphic_matching(&g1, &g2, node_match, edge_match));
    }

    #[test]
    fn test_restrict_left_turns() {
        // these functions used for petgraph::algo::is_isomorphic_matching to test node/edge equality
        let node_match = |n1: &Node, n2: &Node| {n1.eq(n2)};
        let edge_match = |e1: &f32, e2: &f32| {e1==e2};

        let mut orig_graph: Graph<Node, f32, Directed> = Graph::new();
        let modified_graph = prohibit_left_turns2(&orig_graph);
        assert!(is_isomorphic_matching(&orig_graph, &modified_graph, node_match, edge_match));

        // TODO test is_isomorphic_matching to make sure I'm setting up closures correctly
        // TODO write tests for prohibit_left_turns2
    }
}
