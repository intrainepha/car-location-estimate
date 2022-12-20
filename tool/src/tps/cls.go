package tps

type Cls struct {
	Names []string
}

/*
Cls init function

Args:

	ns []string: Slice of class names

Returns:

	*Cls: pointer to a Cls object
*/
func NewCls(ns []string) *Cls {
	return &Cls{Names: ns}
}

/*
Get class ID by class name

Args:

	str string: string data

Returns:

	int: class ID
*/
func (c *Cls) GetID(s string) int {
	for i, v := range c.Names {
		if v == s {
			return i
		}
	}
	return -1
}
