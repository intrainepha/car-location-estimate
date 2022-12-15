package tps

type Rect struct {
	Xtl float64
	Ytl float64
	Xbr float64
	Ybr float64
}

type ScaleBox struct {
	Xc float64
	Yc float64
	W  float64
	H  float64
}

func NewScl(x float64, y float64, w float64, h float64) *ScaleBox {
	/*ScaleBox init function

	Args:
		x(float64): Central x in ratio
		y(float64): Central y in ratio
		w(float64): Width in ratio
		h(float64): Height in ratio

	Returns:
		(*ScaleBox): Pointer to a ScaleBox object
	*/

	return &ScaleBox{Xc: x, Yc: y, W: w, H: h}
}

type Box struct {
	ID   int
	ImSz Size
	Sz   Size
	Rct  Rect
	Scl  ScaleBox
}

func NewSize(w float64, h float64) *Size {
	/*Size init function

	Args:
		w(float64): Width
		h(float64): Height

	Returns:
		(*Size): Pointer to a Size object
	*/

	return &Size{W: w, H: h}
}

func NewRect(xmin float64, ymin float64, xmax float64, ymax float64) *Rect {
	/*Rect init function

	Args:
		xmin(float64): x value of top-left point
		ymin(float64): y value of top-left point
		xmax(float64): x value of bottom-right point
		ymax(float64): y value of bottom-right point

	Returns:
		(*Rect): Pointer to a Rect object
	*/

	return &Rect{Xtl: xmin, Ytl: ymin, Xbr: xmax, Ybr: ymax}
}

func NewBox(id int, r *Rect, s *Size) *Box {
	/*Box init function, takes Rect in, then
	calculate box size and scaled box:
	[x_central, y_cenral, width, height]

	Args:
		id(int): Object class ID
		r(*Rect): Rectangle represent a bounding box
		s(*Sect): Image width and height

	Returns:
		(*Box): Pointer to a Box object
	*/

	box := Box{ID: id, ImSz: *s, Rct: *r}
	// box.Trim()
	// box.Scale()

	return &box
}

func (b *Box) Trim() *Box {
	/*Adjust box value taht out of boundary

	Args:
		None
	Returns:
		b(*Box): Box object after trimed
	*/

	if b.Rct.Xtl < 0 {
		b.Rct.Xtl = 0
	}
	if b.Rct.Ytl < 0 {
		b.Rct.Ytl = 0
	}
	if b.Rct.Xbr >= float64(b.ImSz.W) {
		b.Rct.Xbr = float64(b.ImSz.W) - 1
	}
	if b.Rct.Ybr >= float64(b.ImSz.H) {
		b.Rct.Ybr = float64(b.ImSz.H) - 1
	}

	return b
}

func (b *Box) Scale() *Box {
	/*Calcualte scaled box

	Args:
		None

	Returns:
		b(*Box): Box object after scaled
	*/

	b.Sz.W = b.Rct.Xbr - b.Rct.Xtl + 1
	b.Sz.H = b.Rct.Ybr - b.Rct.Ytl + 1
	b.Scl.Xc = (b.Rct.Xtl + b.Rct.Xbr) / 2 / b.ImSz.W
	b.Scl.Yc = (b.Rct.Ytl + b.Rct.Ybr) / 2 / b.ImSz.H
	b.Scl.W = b.Sz.W / b.ImSz.W
	b.Scl.H = b.Sz.H / b.ImSz.H

	return b
}

func (b *Box) UnScale() *Box {
	/*Calcualte scaled box

	Args:
		None

	Returns:
		b(*Box): Box object after unscaled
	*/

	b.Sz.W = b.Scl.W * b.ImSz.W
	b.Sz.H = b.Scl.H * b.ImSz.H
	b.Rct.Xtl = (b.Scl.Xc*b.ImSz.W*2 - b.Sz.W + 1) / 2
	b.Rct.Ytl = (b.Scl.Yc*b.ImSz.H*2 - b.Sz.H + 1) / 2
	b.Rct.Xbr = b.Rct.Xtl + b.Sz.W
	b.Rct.Ybr = b.Rct.Ytl + b.Sz.H

	return b
}
