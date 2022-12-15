package tps

type Cls struct {
	Names []string
}

func NewCls(ns []string) *Cls {
	/*Cls init function

	Args:
		ns([]string): Slice of class names

	Returns:
		(*Cls): pointer to a Cls object
	*/

	return &Cls{Names: ns}
}

func (c *Cls) GetID(s string) int {
	/*Get class ID by class name

	Args:
		str(string): string data

	Returns:
		(int): class ID
	*/

	for i, v := range c.Names {
		if v == s {
			return i
		}
	}

	return -1
}
