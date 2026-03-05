#include "ObjectCreator.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include "Boxel.h"

ObjectCreator::ObjectCreator() {

}

ObjectCreator::ObjectCreator(Environment* env){
	ObjectCreator::env = env;
}

void ObjectCreator::set_pending_material_scales(const Ref<Matrix <double, 1, Dynamic>>& scales) {
	pending_material_scales = scales;
	material_scales_pending = true;
}

void ObjectCreator::clear_pending_material_scales() {
	clear_internal_material_buffer();
}

void ObjectCreator::clear_internal_material_buffer() {
	pending_material_scales.resize(0, 0);
	material_scales_pending = false;
}

double ObjectCreator::sanitize_material_scale(double raw_scale) const {
	if (!std::isfinite(raw_scale))
		return 1.0;
	double clamped = raw_scale;
	if (clamped < 0.5)
		clamped = 0.5;
	else if (clamped > 2.0)
		clamped = 2.0;
	return clamped;
}

double ObjectCreator::get_boxel_material_scale(int flat_index) const {
	if (!material_scales_pending)
		return 1.0;
	if (flat_index < 0 || flat_index >= pending_material_scales.size())
		return 1.0;
	return sanitize_material_scale(pending_material_scales[flat_index]);
}

void ObjectCreator::reset_edge_material_flags() {
	for (auto& edge : edges) {
		edge.reset_material_flag();
	}
}

int ObjectCreator::get_index(int x, int y) {
	return y * grid_width + x;
}

vector <int> ObjectCreator::get_connected_components(){
	
	vector<bool> is_explored(grid_width*grid_height, false);
	vector<int> component_values(grid_width*grid_height, -1);
	
	int component_id = 0;

	for (int y = 0; y < grid_height; y++) {
		for (int x = 0; x < grid_width; x++) {
			if (!is_explored[get_index(x, y)] && grid.at(get_index(x, y)).cell_type != CELL_EMPTY) {
				explore_grid(component_id, x, y, &is_explored, &component_values);
				component_id += 1;
			}
		}
	}

	return component_values;
}

void ObjectCreator::explore_grid(int parent_value, int x, int y, vector<bool>* is_explored, vector<int>* component_values) {

	int current_index = get_index(x, y);

	if (is_explored->at(current_index))
		return;

	if (grid.at(current_index).cell_type == CELL_EMPTY)
		return;

	component_values->at(current_index) = parent_value;
	is_explored->at(current_index) = true;

	Matrix <bool, 8, 1> neighbors = grid.at(current_index).neighbors;

	if (neighbors[TOP])
		explore_grid(parent_value, x, y - 1, is_explored, component_values);
	if (neighbors[BOT])
		explore_grid(parent_value, x, y + 1, is_explored, component_values);
	if (neighbors[LEFT])
		explore_grid(parent_value, x - 1, y, is_explored, component_values);
	if (neighbors[RIGHT])
		explore_grid(parent_value, x + 1, y, is_explored, component_values);
}

void ObjectCreator::reset() {
	
	grid_width = 0;
	grid_height = 0;
	grid.clear();

	pos.clear();
	vel.clear();
	masses.clear();
	fixed.clear();
	edges.clear();
}

int ObjectCreator::make_point(Vector2d p, Vector2d v, double m, bool f) {
	
	pos.push_back(p);
	vel.push_back(v);
	masses.push_back(m);
	fixed.push_back(f);

	return pos.size() + env->get_num_points() - 1;

}

int ObjectCreator::make_edge(int a_index, int b_index, double length_eq, double spring_const) {

	edges.push_back(Edge(a_index, b_index, length_eq, spring_const));
	// NOTE: make_edge returns a GLOBAL edge index (env offset + local index).
	return edges.size() + env->get_num_edges() - 1;

}

Edge* ObjectCreator::get_edge(int index) {
	// get_edge expects a GLOBAL index; subtract env->get_num_edges() to access local storage.
	return &(edges[index - env->get_num_edges()]);
}

int ObjectCreator::get_point_index(int index) {
	return index - env->get_num_points();
}

void ObjectCreator::init_grid(Vector2d grid_size) {
	
	world_grid_width = grid_size[0];
	world_grid_height = grid_size[1];
	
	for (int i = 0; i < world_grid_width * world_grid_height; i++) {
		world_grid.push_back(0);
	}
}

//wstring exe_path() {
//	TCHAR buffer[MAX_PATH] = { 0 };
//	GetModuleFileName(NULL, buffer, MAX_PATH);
//	return wstring(buffer);
//}

bool ObjectCreator::read_object_from_file(string file_name, string object_name, Vector2d init_pos, bool is_robot) {
	
	Matrix <double, 1, Dynamic> local_grid;
	Matrix <double, 2, Dynamic> connections;
	int grid_width;
	int grid_height;

	reset();

	//ofstream MyFile("THIS_IS_DIRECTORY.txt");
	//MyFile.close();
	//cout << "Reading file from directory:" << std::filesystem::current_path() << "\n";


	ifstream file(file_name);

	if (!file.is_open()) {
		cout << "Error: Could not read in object - file '" << file_name << "' not found.\n";
		return false;
	}
	
	if (!file.good()) {
		cout << "Error: Could not read in object - file '" << file_name << "' not found.\n";
		return false;
	}

	file >> grid_width;
	file >> grid_height;

	local_grid.conservativeResize(grid_width*grid_height);

	//READ BOXELS
	int input;
	for (int i = 0; i < grid_width*grid_height; i++) {
		file >> input;
		local_grid[i] = input;
	}

	//READ CONNECTIONS
	int boxel_index_1;
	int boxel_index_2;
	int num_pairs = 0;

	while (!file.eof()) {
		file >> boxel_index_1 >> boxel_index_2;
		num_pairs++;
		connections.conservativeResize(2, num_pairs);
		connections.col(num_pairs - 1) = Vector2d(boxel_index_1, boxel_index_2);
	}

	file.close();

	return read_object_from_array(object_name, local_grid, connections, Vector2d(grid_width, grid_height), init_pos, is_robot);
}

bool ObjectCreator::is_valid_in_world_grid(int x, int y) {
	
	if (x < 0 || x >= world_grid_width) {
		cout << "Error: Could not read in object - object has cells outside of defined grid area.\n";
		return false;
	}

	if (y < 0 || y >= world_grid_height) {
		cout << "Error: Could not read in object - object has cells outside of defined grid area.\n";
		return false;
	}

	if (world_grid[y * world_grid_width + x] != 0) {
		cout << "Error: Could not read in object - object has overlapping cells with other objects.\n";
		return false;
	}

	return true;
}

bool ObjectCreator::read_object_from_array(string object_name, Matrix <double, 1, Dynamic> local_grid, Matrix <double, 2, Dynamic> connections, Vector2d grid_size, Vector2d init_pos, bool is_robot) {

	reset();

	Vector2d start_pos = init_pos.cwiseProduct(cell_size);
	grid_width = grid_size[0];
	grid_height = grid_size[1];

	double max_y = start_pos.y() + grid_height * cell_size.y();

	//CHECK OVERLAP WITH WORLD GRID
	for (int y = 0; y < grid_height; y++) {
		for (int x = 0; x < grid_width; x++) {
			
			if (local_grid[get_index(x, y)] == 0)
				continue;
			
			if (!is_valid_in_world_grid(x + init_pos[0], world_grid_height - grid_height + y - init_pos[1])) {
				return false;
			}
		}
	}

	//for (int y = 0; y < world_grid_height; y++) {
	//	for (int x = 0; x < world_grid_width; x++) {

	//		cout << world_grid[y * world_grid_width + x];
	//	}
	//	cout << "\n";
	//}
	//cout << "\n";

	//SET BOXELS

	for (int i = 0; i < grid_width*grid_height; i++) {

		grid.push_back(Boxel(local_grid[i], i));
		grid.back().set_material_scale(get_boxel_material_scale(i));
		//if (local_grid[i] == 0)
		//	grid.push_back(Boxel(CELL_EMPTY, i));
		//if (local_grid[i] == 1)
		//	grid.push_back(Boxel(CELL_RIGID, i));
		//if (local_grid[i] == 2)
		//	grid.push_back(Boxel(CELL_SOFT, i));
		//if (local_grid[i] == 3)
		//	grid.push_back(Boxel(CELL_ACT_H, i));
		//if (local_grid[i] == 4)
		//	grid.push_back(Boxel(CELL_ACT_V, i));
	}

	//SET NEIGHBORS

	int boxel_index_1;
	int boxel_index_2;


	for (int i = 0; i < connections.cols(); i++) {

		boxel_index_1 = connections.col(i)[0];
		boxel_index_2 = connections.col(i)[1];

		//Enforce boxel_index_1 <= boxel_index_2
		if (boxel_index_1 > boxel_index_2) {
			int temp = boxel_index_1;
			boxel_index_1 = boxel_index_2;
			boxel_index_2 = temp;
		}

		//horizontal connection
		if (abs(boxel_index_1 - boxel_index_2) == 1 && grid_width != 1) {
			grid[boxel_index_1].neighbors[RIGHT] = true;
			grid[boxel_index_2].neighbors[LEFT] = true;
		}

		//vertical connection
		if (abs(boxel_index_1 - boxel_index_2) == grid_width) {
			grid[boxel_index_1].neighbors[BOT] = true;
			grid[boxel_index_2].neighbors[TOP] = true;
		}
	}

	//CREATE POINTS AND EDGES

	for (int y = 0; y < grid_height; y++) {
		for (int x = 0; x < grid_width; x++) {

			if (grid[get_index(x, y)].cell_type == CELL_EMPTY)
				continue;

			Boxel* current = &grid[get_index(x, y)];

			Boxel* top = NULL;
			Boxel* bot = NULL;
			Boxel* left = NULL;
			Boxel* right = NULL;

			Boxel* top_left = NULL;
			Boxel* top_right = NULL;
			Boxel* bot_left = NULL;
			Boxel* bot_right = NULL;

			//SET CARDINAL POINTS
			if (current->neighbors[TOP])
				top = &grid[get_index(x, y - 1)];
			if (current->neighbors[BOT])
				bot = &grid[get_index(x, y + 1)];
			if (current->neighbors[LEFT])
				left = &grid[get_index(x - 1, y)];
			if (current->neighbors[RIGHT])
				right = &grid[get_index(x + 1, y)];

			//SET DIAGONAL POINTS
			if (current->neighbors[TOP] && top->neighbors[LEFT])
				top_left = &grid[get_index(x - 1, y - 1)];
			if (current->neighbors[LEFT] && left->neighbors[TOP])
				top_left = &grid[get_index(x - 1, y - 1)];

			if (current->neighbors[TOP] && top->neighbors[RIGHT])
				top_right = &grid[get_index(x + 1, y - 1)];
			if (current->neighbors[RIGHT] && right->neighbors[TOP])
				top_right = &grid[get_index(x + 1, y - 1)];

			if (current->neighbors[BOT] && bot->neighbors[LEFT])
				bot_left = &grid[get_index(x - 1, y + 1)];
			if (current->neighbors[LEFT] && left->neighbors[BOT])
				bot_left = &grid[get_index(x - 1, y + 1)];

			if (current->neighbors[BOT] && bot->neighbors[RIGHT])
				bot_right = &grid[get_index(x + 1, y + 1)];
			if (current->neighbors[RIGHT] && right->neighbors[BOT])
				bot_right = &grid[get_index(x + 1, y + 1)];

			if (top_left != NULL)
				current->neighbors[TOP_LEFT] = true;
			if (top_right != NULL)
				current->neighbors[TOP_RIGHT] = true;
			if (bot_left != NULL)
				current->neighbors[BOT_LEFT] = true;
			if (bot_right != NULL)
				current->neighbors[BOT_RIGHT] = true;

			//TOP_LEFT_POINT
			if (top != NULL && top->point_bot_left_index >= 0)
				current->point_top_left_index = top->point_bot_left_index;
			if (left != NULL && left->point_top_right_index >= 0)
				current->point_top_left_index = left->point_top_right_index;
			if (top_left != NULL && top_left->point_bot_right_index >= 0)
				current->point_top_left_index = top_left->point_bot_right_index;
			if (current->point_top_left_index < 0)
			{
				current->point_top_left_index = make_point(
					Vector2d(start_pos.x() + x*cell_size.x(), max_y - y * cell_size.y()),
					Vector2d(0,0), mass, false);
				//current->point_top_left_index = env->createPoint(Vector2d_old(start_pos.x() + x * cell_size.x(), max_y - y * cell_size.y()), mass);
			}

			//TOP_RIGHT_POINT
			if (top != NULL && top->point_bot_right_index >= 0)
				current->point_top_right_index = top->point_bot_right_index;
			if (right != NULL && right->point_top_left_index >= 0)
				current->point_top_right_index = right->point_top_left_index;
			if (top_right != NULL && top_right->point_bot_left_index >= 0)
				current->point_top_right_index = top_right->point_bot_left_index;
			if (current->point_top_right_index < 0)
			{
				current->point_top_right_index = make_point(
					Vector2d(start_pos.x() + (x + 1)*cell_size.x(), max_y - y*cell_size.y()),
					Vector2d(0, 0), mass, false);
				//current->point_top_right_index = env->createPoint(Vector2d_old(start_pos.x() + (x + 1)*cell_size.x(), max_y - y * cell_size.y()), mass);
			}

			//BOT_LEFT_POINT
			if (bot != NULL && bot->point_top_left_index >= 0)
				current->point_bot_left_index = bot->point_top_left_index;
			if (left != NULL && left->point_bot_right_index >= 0)
				current->point_bot_left_index = left->point_bot_right_index;
			if (bot_left != NULL && bot_left->point_top_right_index >= 0)
				current->point_bot_left_index = bot_left->point_top_right_index;
			if (current->point_bot_left_index < 0)
			{
				current->point_bot_left_index = make_point(
					Vector2d(start_pos.x() + x*cell_size.x(), max_y - (y + 1)*cell_size.y()),
					Vector2d(0, 0), mass, false);
				//current->point_bot_left_index = env->createPoint(Vector2d_old(start_pos.x() + x * cell_size.x(), max_y - (y + 1)*cell_size.y()), mass);
				
			}

			//BOT_RIGHT_POINT
			if (bot != NULL && bot->point_top_right_index >= 0)
				current->point_bot_right_index = bot->point_top_right_index;
			if (right != NULL && right->point_bot_left_index >= 0)
				current->point_bot_right_index = right->point_bot_left_index;
			if (bot_right != NULL && bot_right->point_top_left_index >= 0)
				current->point_bot_right_index = bot_right->point_top_left_index;
			if (current->point_bot_right_index < 0)
			{
				current->point_bot_right_index = make_point(
					Vector2d(start_pos.x() + (x + 1)*cell_size.x(), max_y - (y + 1)*cell_size.y()),
					Vector2d(0, 0), mass, false);
				//current->point_bot_right_index = env->createPoint(Vector2d_old(start_pos.x() + (x + 1)*cell_size.x(), max_y - (y + 1)*cell_size.y()), mass);

			}

			//SET SHARED EDGES

			//TOP (unassigned sentinel is -1; do not treat 0 as "unset")
			if (top != NULL && top->edge_bot_index >= 0)
				current->edge_top_index = top->edge_bot_index;
			if (current->edge_top_index < 0) {
				current->edge_top_index = make_edge(current->point_top_left_index, current->point_top_right_index, abs(cell_size.x()), rigid_main_edge_spring_const);
			}

			//BOT
			if (bot != NULL && bot->edge_top_index >= 0)
				current->edge_bot_index = bot->edge_top_index;
			if (current->edge_bot_index < 0) {
				current->edge_bot_index = make_edge(current->point_bot_left_index, current->point_bot_right_index, abs(cell_size.x()), rigid_main_edge_spring_const);
			}

			//LEFT
			if (left != NULL && left->edge_right_index >= 0)
				current->edge_left_index = left->edge_right_index;
			if (current->edge_left_index < 0) {
				current->edge_left_index = make_edge(current->point_top_left_index, current->point_bot_left_index, abs(cell_size.y()), rigid_main_edge_spring_const);
			}

			//RIGHT
			if (right != NULL && right->edge_left_index >= 0)
				current->edge_right_index = right->edge_left_index;
			if (current->edge_right_index < 0) {
				current->edge_right_index = make_edge(current->point_top_right_index, current->point_bot_right_index, abs(cell_size.y()), rigid_main_edge_spring_const);
			}

			//SET CROSS EDGES
			/*double diagonal_edge_length = sqrt(cell_size.x()*cell_size.x() + cell_size.y()*cell_size.y());
			env->create_edge(Edge(current->point_top_right_index, current->point_bot_left_index, diagonal_edge_length, other_edge_spring_const));
			env->create_edge(Edge(current->point_top_left_index, current->point_bot_right_index, diagonal_edge_length, other_edge_spring_const));*/
		}
	}

	//CUSTOMIZE EDGES
	double diagonal_edge_length = sqrt(cell_size.x()*cell_size.x() + cell_size.y()*cell_size.y());
	reset_edge_material_flags();

	for (int y = 0; y < grid_height; y++) {
		for (int x = 0; x < grid_width; x++) {

			Boxel* current = &grid[get_index(x, y)];
			int cell_type = current->cell_type;
			if (cell_type == CELL_EMPTY)
				continue;

			double base_main = 0.0;
			double base_diag = 0.0;
			bool has_material = true;
			switch (cell_type) {
			case CELL_RIGID:
			case CELL_FIXED:
				base_main = rigid_main_edge_spring_const;
				base_diag = rigid_structural_edge_spring_const;
				break;
			case CELL_SOFT:
				base_main = soft_main_edge_spring_const;
				base_diag = soft_structural_edge_spring_const;
				break;
			case CELL_ACT_H:
			case CELL_ACT_V:
				base_main = actuator_main_edge_spring_const;
				base_diag = actuator_structural_edge_spring_const;
				break;
			default:
				has_material = false;
				break;
			}

			if (!has_material)
				continue;

			double scale = sanitize_material_scale(current->get_material_scale());
			double k_main = base_main * scale;
			double k_diag = base_diag * scale;

			if (current->edge_top_index >= 0)
				get_edge(current->edge_top_index)->assign_material_k(k_main);
			if (current->edge_bot_index >= 0)
				get_edge(current->edge_bot_index)->assign_material_k(k_main);
			if (current->edge_left_index >= 0)
				get_edge(current->edge_left_index)->assign_material_k(k_main);
			if (current->edge_right_index >= 0)
				get_edge(current->edge_right_index)->assign_material_k(k_main);

			make_edge(current->point_top_right_index, current->point_bot_left_index, diagonal_edge_length, k_diag);
			make_edge(current->point_top_left_index, current->point_bot_right_index, diagonal_edge_length, k_diag);
		}
	}

	// DUMPK: build-time dump of per-voxel scales and main-edge spring constants (robot only)
	// #if 0
	// if (object_name == "robot") {
	// 	cout << "[DumpK] name=" << object_name << " grid=(" << grid_width << "," << grid_height << ")\n";
	// 	for (int y = 0; y < grid_height; y++) {
	// 		for (int x = 0; x < grid_width; x++) {
	// 			Boxel* current = &grid[get_index(x, y)];
	// 			if (current->cell_type == CELL_EMPTY)
	// 				continue;

	// 			cout << "Voxel(x=" << x << ", y=" << y << ") type=" << current->cell_type << " s=" << current->get_material_scale() << "\n";

	// 			auto print_k = [&](const char* label, int edge_index) {
	// 				int local_start = env->get_num_edges();
	// 				int local_end = env->get_num_edges() + (int)edges.size();
	// 				if (edge_index < local_start || edge_index >= local_end) {
	// 					cout << label << "N/A";
	// 					return;
	// 				}
	// 				cout << label << get_edge(edge_index)->spring_const;
	// 			};

	// 			cout << "  ";
	// 			print_k("k_top=", current->edge_top_index);
	// 			cout << " ";
	// 			print_k("k_bottom=", current->edge_bot_index);
	// 			cout << " ";
	// 			print_k("k_left=", current->edge_left_index);
	// 			cout << " ";
	// 			print_k("k_right=", current->edge_right_index);
	// 			cout << "\n";
	// 		}
	// 	}
	// }
	// #endif

	//COMPUTE FIXED
	for (int y = 0; y < grid_height; y++) {
		for (int x = 0; x < grid_width; x++) {

			Boxel* current = &grid[get_index(x, y)];

			if (current->cell_type != CELL_FIXED)
				continue;

			fixed[get_point_index(current->point_top_left_index)] = true;
			fixed[get_point_index(current->point_top_right_index)] = true;
			fixed[get_point_index(current->point_bot_left_index)] = true;
			fixed[get_point_index(current->point_bot_right_index)] = true;
		}
	}

	//ESTABLISH OBJECTS BY RUNNING FLOOD FILL
	vector<int> connected_components = get_connected_components();
	
	int highest_component_tracker = -1;
	int current_index;

	int num_objects_already_added = env->get_objects()->size();

	//ENFORCE ONLY ONE CONNECTED COMPONENT
	for (int y = 0; y < grid_height; y++) {
		for (int x = 0; x < grid_width; x++) {

			current_index = get_index(x, y);

			if (connected_components[current_index] >= 1) {
				cout << "Error: Could not read in object - object must be connected.\n";
				return false;
			}

		}
	}

	//CHECK & ADD NAME
	if (!env->add_object_name(object_name, env->get_objects()->size())) {
		if (is_robot)
			cout << "Error Could not read in robot " << object_name << " - duplicate name.\n";
		else
			cout << "Error Could not read in object " << object_name << " - duplicate name.\n";
		return false;
	}
	
	//CREATE POINTS & EDGES
	int min_point_index = env->get_pos()->cols();

	env->create_points(&pos, &vel, &masses, &fixed);
	env->create_edges(&edges);

	int max_point_index = env->get_pos()->cols() - 1;

	//UPDATE WORLD GRID
	for (int y = 0; y < grid_height; y++) {
		for (int x = 0; x < grid_width; x++) {

			if (local_grid[get_index(x, y)] == 0)
				continue;

			world_grid[(world_grid_height - grid_height + y - init_pos[1]) * world_grid_width + (x + init_pos[0])] = local_grid[get_index(x, y)];
		}
	}

	//SET OBJECTS
	for (int y = 0; y < grid_height; y++) {
		for (int x = 0; x < grid_width; x++) {

			current_index = get_index(x, y);

			if (connected_components[current_index] == -1)
				continue;

			if (connected_components[current_index] > highest_component_tracker) {
				if (is_robot)
				{
					env->get_objects()->push_back(new Robot());
					env->get_objects()->at(env->get_objects()->size()-1)->is_robot = true;
				}
				else
					env->get_objects()->push_back(new SimObject());

				highest_component_tracker = connected_components[current_index];
			}

			// select env object at index of connected component. To that add grid boxel to boxel array.
			env->get_objects()->at(connected_components[current_index] + num_objects_already_added)->boxels.push_back(grid[current_index]);

		}
	}

	//MIN MAX POINTS INDEX
	for (int i = num_objects_already_added; i < env->get_objects()->size(); i++) {
		env->get_objects()->at(i)->max_point_index = max_point_index;
		env->get_objects()->at(i)->min_point_index = min_point_index;
	}

	//INIT BOXELS
	for (int i = num_objects_already_added; i < env->get_objects()->size(); i++) {
		for (int j = 0; j < env->get_objects()->at(i)->boxels.size(); j++) {
			env->get_objects()->at(i)->boxels.at(j).init();
		}
	}

	//COMPUTE SURFACES
	for (int i = num_objects_already_added; i < env->get_objects()->size(); i++) {
		
		SimObject* current_obj = env->get_objects()->at(i);

		current_obj->compute_surface();
		
		for (int j = 0; j < current_obj->surface_edges_index.cols(); j++) {

			Edge* current_edge = &env->get_edges()->at(current_obj->surface_edges_index[j]);
			int type = current_obj->surface_edge_directions[current_obj->surface_edges_index[j]];

			Vector2d diff = env->get_pos()->col(current_edge->b_index) - env->get_pos()->col(current_edge->a_index);
			Vector2d norm = Vector2d(-diff.y(), diff.x());

			Vector2d ref;
			if (type == TOP)
				ref = Vector2d(0.0, 1.0);
			if (type == BOT)
				ref = Vector2d(0.0, -1.0);
			if (type == LEFT)
				ref = Vector2d(-1.0, 0.0);
			if (type == RIGHT)
				ref = Vector2d(1.0, 0.0);


			double dot = ref.dot(norm); //ref.x() * norm.x() + ref.y() + norm.y();

			if (dot < 0)
				env->swap_edge(current_obj->surface_edges_index[j]);
		}

		current_obj->compute_bb_tree(*env->get_pos(), grid_width, grid_height);
		
	}

	env->set_surface_edge_color(2);
	clear_internal_material_buffer();

	return true;
}

ObjectCreator::~ObjectCreator()
{
}
