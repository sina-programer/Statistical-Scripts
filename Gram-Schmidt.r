dot = function(x, y){
  sum(x * y)
}

norm = function(vector){
  sqrt(dot(vector, vector))
}

unitize = function(vector){
  vector / norm(vector)
}

project = function(u, v){
	a = dot(u, v) / dot(v, v)
	v * a
}

orthogonalize = function(v1, v2){
	u1 = v1
	u2 = v2 - project(v2, u1)
	list(u1=u1, u2=u2)
}

orthonormalize = function(v1, v2):
	U = orthogonalize(v1, v2)
	list(u1=unitize(U$u1), u2=unitize(U$u2))

