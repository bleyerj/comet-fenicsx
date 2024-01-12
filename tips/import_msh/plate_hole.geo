// Define geometry parameters
L = 10;  // Side length of the square plate
R = 2;   // Radius of the circular hole

// Define points
Point(1) = {0, 0, 0, 1.0};        // Center of the square
Point(2) = {-L/2, -L/2, 0, 1.0};  // Bottom-left corner of the square
Point(3) = {L/2, -L/2, 0, 1.0};   // Bottom-right corner of the square
Point(4) = {L/2, L/2, 0, 1.0};    // Top-right corner of the square
Point(5) = {-L/2, L/2, 0, 1.0};   // Top-left corner of the square

Point(6) = {0, 0, 0, 1.0};        // Center of the circle
Point(7) = {R, 0, 0, 1.0};        // 1st point on the circle
Point(8) = {0, R, 0, 1.0};        // 2nd point on the circle
Point(9) = {-R, 0, 0, 1.0};        // 3rd point on the circle
Point(10) = {0, -R, 0, 1.0};        // 4th point on the circle

// Define lines
Line(1) = {2, 3};  // Bottom side of the square
Line(2) = {3, 4};  // Right side of the square
Line(3) = {4, 5};  // Top side of the square
Line(4) = {5, 2};  // Left side of the square

// Define a circle
Circle(5) = {7, 6, 8};  // Circle arcs
Circle(6) = {8, 6, 9};  // 
Circle(7) = {9, 6, 10};  // 
Circle(8) = {10, 6, 7};  // 

// Define the loop (square with a hole)
Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {5, 6, 7, 8};
Plane Surface(1) = {9, 10};

// Physical entities
Physical Surface(1) = {1};

Physical Line(10) = {1,2,3,4};
Physical Line(20) = {5,6,7,8};

// Meshing parameters
Mesh.CharacteristicLengthMin = 0.5;
Mesh.CharacteristicLengthMax = 2.0;
