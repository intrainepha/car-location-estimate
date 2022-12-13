package types

type Location struct {
	X float64
	Y float64
}

func NewLoc(x float64, y float64) *Location {
	/*Location init function

	Args:
		x(float64): x-distance in real world
		y(float64): y-distance in real world

	Returns:
		(*Location): Pointer to a Location object
	*/

	return &Location{X: x, Y: y}
}
