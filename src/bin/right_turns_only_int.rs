use nannou::draw::properties::Vertices;
use nannou::prelude::*;
use ordered_float::OrderedFloat;
use osmpbfreader::{Node, Way, NodeId};
use petgraph::graph::NodeIndex;
use petgraph::graph::*;
use petgraph::prelude::*;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;

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

const MAX_LON: i32 = -713621000;
const MIN_LON: i32 = -713820000;
const MAX_LAT: i32 =  415028000;
const MIN_LAT: i32 =  414938000;

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

const LON_RANGE: i32 = MAX_LON - MIN_LON;
const LAT_RANGE: i32 = MAX_LAT - MIN_LAT;

const MAX_WIN_WIDTH: f32 = 1357.0 * 0.7; // based on my laptop screen
const MAX_WIN_HEIGHT: f32 = 657.0 * 0.7; // based on my laptop screen

const WIN_W: f32 = ((MAX_WIN_HEIGHT as i32) / LAT_RANGE * LON_RANGE) as f32;
const WIN_H: f32 = MAX_WIN_HEIGHT;

const HUE_MAX: f32 = 0.9; // the nannou API for hsv colors takes an f32 between 0.0 and 1.0
const HUE_MIN: f32 = 0.55;

fn main() {
    /*
    figure out how to build a OSM Node
    tests will be done by outputting to the console every edge in the graph, and I will manually check they are correct
    Test on following options:
        - empty graph
        - one node graph
        - two nodes connected to each other
        - one four way
        - a four way connected to an extension
        - two four ways connected to eachother
        - (if possible) three four ways connected in a triangle
    */
    /*
    let node0: Node = Node{ id: NodeId(0), tags: Default::default(), decimicro_lat: 0, decimicro_lon: 0};
    let node1: Node = Node{ id: NodeId(1), tags: Default::default(), decimicro_lat: -10, decimicro_lon: 1};
    let node2: Node = Node{ id: NodeId(2), tags: Default::default(), decimicro_lat: 1, decimicro_lon: 10};
    let node3: Node = Node{ id: NodeId(3), tags: Default::default(), decimicro_lat: 10, decimicro_lon: -1};
    let node4: Node = Node{ id: NodeId(4), tags: Default::default(), decimicro_lat: -1, decimicro_lon: -10};
    let node5: Node = Node{ id: NodeId(5), tags: Default::default(), decimicro_lat: 10, decimicro_lon: 10};
    let node6: Node = Node{ id: NodeId(6), tags: Default::default(), decimicro_lat: 20, decimicro_lon: -1};
    let node7: Node = Node{ id: NodeId(7), tags: Default::default(), decimicro_lat: 10, decimicro_lon: -10};
    let mut test_graph = Graph::<Node, f32, Directed>::new();

    let n0 = test_graph.add_node(node0);
    let n1 = test_graph.add_node(node1);
    let n2 = test_graph.add_node(node2);
    let n3 = test_graph.add_node(node3);
    let n4 = test_graph.add_node(node4);
    let n5 = test_graph.add_node(node5);
    let n6 = test_graph.add_node(node6);
    let n7 = test_graph.add_node(node7);
    test_graph.add_edge(n0, n1, 1.0);
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

    prohibit_left_turns(&mut test_graph);
    println!("NODE COUNT: {}", test_graph.node_count());
    println!("EDGE COUNT: {}", test_graph.edge_count());
    let mut stupid_hashmap: HashMap<NodeIndex, Vec<(NodeIndex,NodeIndex)>> = HashMap::new();
    for edge in test_graph.raw_edges() {
        let edge_endpoints = (edge.source(), edge.target());
        let prior_edges = stupid_hashmap.get_mut(&edge.source());
        if prior_edges.is_some() {
            prior_edges.unwrap().push(edge_endpoints);
        } else {
            stupid_hashmap.insert(edge.source(), vec!(edge_endpoints));
        }

    }
    for node_ix in test_graph.node_indices() {
        // print node index and the coordinates (petgraph) (maybe pre-do this so it's a list above)
        let node = test_graph.node_weight(node_ix).unwrap();
        println!("NodeIndex: {}, located at ({},{})", node_ix.index(), node.decimicro_lat, node.decimicro_lon);
        if stupid_hashmap.contains_key(&node_ix) {
            for edge in stupid_hashmap.get(&node_ix).unwrap() {
                println!("\te: ({},{})", edge.0.index(), edge.1.index());
            }
        }

    }*/

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

    let mut test_node = NodeIndex::new(1);

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
    let mut road_graph = Graph::<Node, i32, Directed>::new();
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
                        test_node = cur_node_index;
                    }
                    graph_node_indices.insert(node, cur_node_index);
                }

                // if it's not the first one, form an edge along the path
                if i != 0 {
                    // find distances between the two points
                    let prior_node = road_graph
                        .node_weight(prior_node_index)
                        .expect("prior node should exist because we already traversed it");
                    let start_point = pt2(prior_node.decimicro_lon, prior_node.decimicro_lat);
                    let end_point = pt2(node.decimicro_lon, node.decimicro_lat);

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





    prohibit_left_turns(&mut road_graph);
    let road_lines: Vec<Line> = color_roads(&road_graph, test_node);

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
Nannou runs function 60 times per second to produce a new frame.
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
    (node.decimicro_lon < MAX_LON)
        & (node.decimicro_lon > MIN_LON)
        & (node.decimicro_lat < MAX_LAT)
        & (node.decimicro_lat > MIN_LAT)
}

fn is_in_outer_bounds(node: &Node) -> bool {
    let outer_max_lon = MAX_LON + LON_RANGE * 1;
    let outer_min_lon = MIN_LON - LON_RANGE * 1;
    let outer_max_lat = MAX_LAT + LAT_RANGE * 1;
    let outer_min_lat = MIN_LAT - LAT_RANGE * 1;

    (node.decimicro_lon < outer_max_lon)
        & (node.decimicro_lon > outer_min_lon)
        & (node.decimicro_lat < outer_max_lat)
        & (node.decimicro_lat > outer_min_lat)
}

fn dist(pt1: Point2<i32>, pt2: Point2<i32>) -> i32 {
    ((pt1.x - pt2.x).powi(2) + (pt1.y - pt2.y).powi(2)).sqrt()
}

fn convert_coord(node: &Node) -> Point2 {
    let x = map_range(node.decimicro_lon, MIN_LON, MAX_LON, -WIN_W * 0.5, WIN_W * 0.5);
    let y = map_range(node.decimicro_lat, MIN_LAT, MAX_LAT, -WIN_W * 0.5, WIN_H * 0.5);
    pt2(x, y)
}

fn color_roads(road_graph: &Graph<Node, i32, Directed>, test_node: NodeIndex) -> Vec<Line> {
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

// stores the data I want to use when I draw a line using nannou's draw.line() builder, namely the end points and values for the color/thickness
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
) -> HashMap<NodeIndex, Option<i32>, RandomState> {
    if g.node_weight(start).is_some() {
        let mut min_dist: HashMap<NodeIndex, i32, RandomState>; // final return value; maps node indices to their min distance from the start
        let mut edge_dist: BinaryHeap<DistFloat>; // potential edges to explore; each Dist stores a node and a distance leading up to that node (which may not be the min distance)
                                                  // Djikstra's formula works by always selecting the minimum of those potential edges (hence why it's a BinaryHeap (Priority Queue)

        // initialize min_dist
        min_dist = HashMap::new();
        for node in g.node_indices() {
            min_dist.insert(node, std::i32::INFINITY);
        }
        min_dist.insert(start, 0.0);

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

        let min_dist: HashMap<NodeIndex, Option<i32>, RandomState> = min_dist
            .into_iter()
            .map(|(node, dist)| {
                if dist == std::i32::INFINITY {
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

//Mutates road graph so 4 way intersections don't allow left turns.
fn right_turns_only(g: &mut Graph<Node, i32, Directed>) {
    // in this hashmap, the key is the original 4 way intersection, the value is list of nodes representing 4 separate directions
    let mut directional_nodes: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();

    for node_ix in g.node_indices() {
        if g.neighbors(node_ix).count() == 4 {
            // MAKE 4 NEW NODES that store clone of the original node's weight, set each one to a different direction
            let mut neighbors: Vec<NodeIndex> = g.neighbors(node_ix).collect();
            neighbors = get_clockwise_order(&g, node_ix, neighbors);
            let mut neighbors_iter = neighbors.into_iter();
            let mut four_copies = Vec::<NodeIndex>::new();

            for i in 0..4 {
                // make copy of node's stored value (its weight)
                let node_weight = g.node_weight(node_ix).unwrap().clone();
                // get next neighbor as well as the edge weight to that neighbor
                let neighbor = neighbors_iter.next().unwrap();
                let prior_edge = g.find_edge(node_ix, neighbor).unwrap();
                let edge_weight = *g.edge_weight(prior_edge).unwrap();

                // create new node and set it's outgoing edge to the outgoing edge (this will get changed later so it connects to the write 4-direction node)
                let new_node = g.add_node(node_weight);
                g.add_edge(new_node, neighbor, edge_weight);
                // remove prior edges (in both directions)
                g.remove_edge(prior_edge);
                let edge_other_direction = g.find_edge(neighbor, node_ix).unwrap();
                g.remove_edge(edge_other_direction);
                //
                four_copies.push(new_node);
            }

            // add four directions to the directional nodes hashmap
            // note that here the list of nodes should be in clockwise order of the directions of their edges
            directional_nodes.insert(node_ix, four_copies);
        }
    }

    // after we've set all the outgoing edges (and all 4 way intersections have the 4 clones), we can now align them so they come in to the correct node)
    for (node, clones) in directional_nodes.iter() {
        // for each incoming node to the original node, align it with the correct clone node
        // note that at this point there shouldn't be redundancy because we deleted the original edges after we added the cloned one
        // for each incoming edge:
        // find which clone node is the opposite direction
        // from there, find the ones that go straight and turn right
        // connect!
    }
}

/**
Returns neighbors of graph node in clockwise order by compass directions (which direction starts is whatever's closest to west)
*/
fn get_clockwise_order(
    g: &Graph<Node, i32, Directed>,
    center: NodeIndex,
    neighbors: Vec<NodeIndex>,
) -> Vec<NodeIndex> {
    let center = g.node_weight(center).unwrap();
    let mut neighbor_angles: Vec<(f64, NodeIndex)> = neighbors
        .into_iter()
        .map(|neighbor_node| {
            let node = g.node_weight(neighbor_node).unwrap();
            let dx = node.lon() - center.lon();
            let dy = node.lat() - center.lat();
            let angle = dy.atan2(dx);
            (angle, neighbor_node)
        })
        .collect();

    neighbor_angles.sort_by(|a, b| (-a.0).partial_cmp(&(-b.0)).unwrap());
    neighbor_angles
        .into_iter()
        .map(|(_angle, node_ix)| node_ix)
        .collect()
}

/**
Takes pairings of neighbors/internal nodes and returns them sorted such that the neighbors are arranged clockwise around the given center (starting with whichever is in the northwest quadrant closest to west)
*/
fn get_clockwise_order2(
    g: &Graph<Node, i32, Directed>,
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


/**
Changes the graph so that you cannot make left turns while traversing the graph.

For each four way intersection, it creates 4 clones representing the 4 different directions you could leave that intersection. Then you can make all incoming edges to that intersection point to only those clones that are legal for them (i.e. the straight and right directions). Finally, the original nodes are removed (so the original nodes can't be used after this --> perhaps I should change this and have it create a new graph entirely?)
*/
fn prohibit_left_turns(g: &mut Graph<Node, i32, Directed>) {
    let clones = create_clones(g);
    let outgoing_edges = add_outgoing_edges(g, clones);
    let intersections = make_intersection_objs(g, outgoing_edges);
    add_incoming_edges(g, &intersections);
    delete_orig_nodes(g, &intersections);
}

/**
For each node in the graph that represents a 4 way intersection, this adds 4 clones to the graph (but with no edges yet) and returns a mapping from the original node to these clones.
*/
fn create_clones(
    g: &mut Graph<Node, i32, Directed>,
) -> HashMap<NodeIndex, Vec<NodeIndex>, RandomState> {
    let mut orig_mapped_to_clones = HashMap::new();
    for node_ix in g.node_indices() {
        // find the nodes in the original graph that have 4 neighbors (i.e. represent a 4 way intersection)
        if g.neighbors(node_ix).count() == 4 {
            let mut clones = Vec::new();
            // add 4 clones of the node to the graph and then store in the HashMap
            for i in 0..4 {
                let node_weight = g.node_weight(node_ix).unwrap().clone();
                clones.push(g.add_node(node_weight));
            }
            orig_mapped_to_clones.insert(node_ix, clones);
        }
    }
    orig_mapped_to_clones
}

/**
Once you have the clones, then you align them so each one points to a different neighbor node (remember, these "clones" represent the 4 different directions you could go to from the intersection)
*/
fn add_outgoing_edges(
    g: &mut Graph<Node, i32, Directed>,
    orig_mapped_to_clones: HashMap<NodeIndex, Vec<NodeIndex>, RandomState>,
) -> HashMap<NodeIndex, Vec<EdgeIndex>> {
    let mut outgoing_edges = HashMap::new();
    for (node, clones) in orig_mapped_to_clones {
        let mut edges_from_clones = Vec::new();
        // for each neighbor of the original node, get its edge weight and recreate that edge from one of the clones to that neighbor
        let mut clones_iter = clones.into_iter();
        let neighbors: Vec<NodeIndex> = g.neighbors(node).collect();
        for neighbor in neighbors.into_iter() {
            let orig_edge = g.find_edge(node, neighbor).unwrap();
            let edge_weight = *g.edge_weight(orig_edge).unwrap();
            let clone: NodeIndex = clones_iter.next().unwrap();
            let new_edge = g.add_edge(clone, neighbor, edge_weight);
            edges_from_clones.push(new_edge);
        }
        outgoing_edges.insert(node, edges_from_clones);
    }
    outgoing_edges
}

/**
Takes the edges leaving the clones and converts the into FourWay objects that can be easily queried regarding which direction to connect to
*/
fn make_intersection_objs(
    g: &mut Graph<Node, i32, Directed>,
    edges: HashMap<NodeIndex, Vec<EdgeIndex>>,
) -> HashMap<NodeIndex, FourWay> {
    let four_way_intersections: HashMap<NodeIndex,FourWay> = edges.into_iter()
        .map(|(orig_node, edges)| {
            (orig_node,FourWay::new(g,orig_node, edges))
        })
        .collect();
    four_way_intersections
}

fn add_incoming_edges(
    g: &mut Graph<Node, i32, Directed>,
    intersections: &HashMap<NodeIndex, FourWay>,
) {
    for (orig_node, clones) in intersections {
        let neighbors: Vec<NodeIndex> = g.neighbors(*orig_node).collect();
        for neighbor in neighbors.into_iter() {
            let orig_edge = g.find_edge(neighbor,*orig_node).unwrap();
            let edge_weight = *g.edge_weight(orig_edge).unwrap();
            let straight_across = clones.get_straight_across(neighbor).unwrap();
            let right = clones.get_right_turn(neighbor).unwrap();
            if (intersections.contains_key(&neighbor)) {
                // handle case where the neighbor is a FourWay intersection
                // if so, you have to fetch the correct clone from that neighbor's FourWay intersection object
                let neighbor_clone = intersections.get(&neighbor).unwrap().get_back(*orig_node).unwrap();
                g.add_edge(neighbor_clone,straight_across,edge_weight);
                g.add_edge(neighbor_clone,right,edge_weight);
            } else {
                // handle case where the neighbor is not an intersection (simply connect it)
                g.add_edge(neighbor,straight_across,edge_weight);
                g.add_edge(neighbor,right,edge_weight);
            }
        }
    }
}

fn delete_orig_nodes(
    g: &mut Graph<Node, i32, Directed>,
    intersections: &HashMap<NodeIndex, FourWay>,
) {
    for (orig_node, _) in intersections {
        g.remove_node(*orig_node);
    }
}

/**
Represents a four way intersection and allows for easy querying about which internal node to connect to given some neighbor node.

Used while building a graphs that limit certain kinds of turning (like making right turn only). The vector contains 4 tuples. Each tuple represents (1) a neighbor node from the intersection and (2) the internal node of the intersection that corresponds to going in the direction of that neighbor

*/
struct FourWay {
    neighbors: Vec<(NodeIndex, NodeIndex)>,
}

impl FourWay {
    pub fn new(
        g: &Graph<Node, i32, Directed>,
        center: NodeIndex,
        neighbors: Vec<EdgeIndex>,
    ) -> FourWay {
        let neighbors: Vec<(NodeIndex, NodeIndex)> = neighbors
            .into_iter()
            .map(|edge_ix| {
                let endpoints = g.edge_endpoints(edge_ix).unwrap();
                (endpoints.1, endpoints.0)
            })
            .collect();
        let neighbors = get_clockwise_order2(g, center, neighbors);
        FourWay { neighbors }
    }

    /**
    Given a neighbor of this intersection, it returns the internal node representing going straight (if coming from that neighbor). If the argument supplied isn't actually one of the neighbors stored in the FourWayIntersection, this will panic.
    */
    pub fn get_straight_across(&self, neighbor: NodeIndex) -> Option<NodeIndex> {
        println!("neighbor is {}", neighbor.index());
        for neighbor in self.neighbors.iter() {
            println!("Neighbor: {}, Internal node: {}", neighbor.0.index(), neighbor.1.index());
        }
        let neighbor_ix = self
            .neighbors
            .iter()
            .position(|x| x.0.eq(&neighbor))
            .unwrap();
        let straight_across_ix = (neighbor_ix + 2) % 4;
        Some(self.neighbors[straight_across_ix].1)
    }

    /**
    Same as get_straight_across but for left turns.
    */
    pub fn get_left_turn(&self, n: NodeIndex) -> Option<NodeIndex> {
        let index = self.neighbors.iter().position(|x| x.0.eq(&n)).unwrap();
        let left_ix = (index + 1) % 4;
        Some(self.neighbors[left_ix].1)
    }

    /**
    Same as get_straight_across but for right turns
    */
    pub fn get_right_turn(&self, n: NodeIndex) -> Option<NodeIndex> {
        let index = self.neighbors.iter().position(|x| x.0.eq(&n)).unwrap();
        let right_ix = (index + 3) % 4;
        Some(self.neighbors[right_ix].1)
    }

    pub fn get_back(&self, n: NodeIndex) -> Option<NodeIndex> {
        let index = self.neighbors.iter().position(|x| x.0.eq(&n)).unwrap();
        Some(self.neighbors[index].1)
    }
}