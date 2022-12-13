package types

type Size struct {
	W float64
	H float64
}

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

type Box struct {
	ImgSz Size
	Sz    Size
	Rct   Rect
	Scl   ScaleBox
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

func NewBox(r *Rect, s *Size) *Box {
	/*Box init function, takes Rect in, then
	calculate box size and scaled box:
	[x_central, y_cenral, width, height]

	Args:
		r(*Rect): Rectangle represent a bounding box
		s(*Sect): Image width and height

	Returns:
		(*Box): Pointer to a Box object
	*/

	box := Box{ImgSz: *s, Rct: *r}
	box.trim()
	box.bale()

	return &box
}

func (b *Box) trim() {
	/*Adjust box value taht out of boundary

	Args:
		None
	Returns:
		None
	*/

	if b.Rct.Xtl < 0 {
		b.Rct.Xtl = 0
	}
	if b.Rct.Ytl < 0 {
		b.Rct.Ytl = 0
	}
	if b.Rct.Xbr >= float64(b.ImgSz.W) {
		b.Rct.Xbr = float64(b.ImgSz.W) - 1
	}
	if b.Rct.Ybr >= float64(b.ImgSz.H) {
		b.Rct.Ybr = float64(b.ImgSz.H) - 1
	}
}

func (b *Box) bale() {
	/*Calcualte scaled box

	Args:
		None

	Returns:
		None
	*/

	b.Sz.W = b.Rct.Xbr - b.Rct.Xtl + 1
	b.Sz.H = b.Rct.Ybr - b.Rct.Ytl + 1
	b.Scl.Xc = (b.Rct.Xtl + b.Rct.Xbr) / 2 / b.ImgSz.W
	b.Scl.Yc = (b.Rct.Ytl + b.Rct.Ybr) / 2 / b.ImgSz.H
	b.Scl.W = b.Sz.W / b.ImgSz.W
	b.Scl.H = b.Sz.H / b.ImgSz.H
}
