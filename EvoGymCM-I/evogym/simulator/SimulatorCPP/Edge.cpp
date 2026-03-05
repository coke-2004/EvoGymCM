#include "Edge.h"

Edge::Edge(int a_index, int b_index, double length_eq, double spring_const, int color)
{
	Edge::a_index = a_index;
	Edge::b_index = b_index;
	Edge::init_length_eq = length_eq;
	Edge::spring_const = spring_const;
	Edge::color = color;
	Edge::isColliding = false;
	Edge::num_actuating = 0;
	Edge::act_length_eq = init_length_eq;
	Edge::isOnSurface = false;
	Edge::has_material_k = false;
}

Vector2d Edge::get_normal(Ref< Matrix <double, 2, Dynamic>> pos) {
	
	Vector2d slope = pos.col(b_index) - pos.col(a_index);
	Vector2d normal = Vector2d(-slope.y(), slope.x());
	
	return normal.normalized();
}

void Edge::swap() {
	int temp = a_index;
	a_index = b_index;
	b_index = temp;
}

void Edge::assign_material_k(double new_k) {
	if (!has_material_k) {
		spring_const = new_k;
		has_material_k = true;
		return;
	}
	spring_const = (spring_const + new_k) * 0.5;
}

void Edge::reset_material_flag() {
	has_material_k = false;
}

Edge::~Edge()
{
}
