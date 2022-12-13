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

func NewBox(r *Rect, s *Size) *Box {
	/*Box init function, takes Rect in, then
	calculate box size and scaled box:
	[x_central, y_cenral, width, height]

	Args:
		r(Rect): Rectangle represent a bounding box
		s(Sect): Image width and height

	Returns:
		*Box: Point to a Box object
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
