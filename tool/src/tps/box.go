package tps

type Rect struct {
	Xtl, Ytl, Xbr, Ybr int
}

type ScaleBox struct {
	Xc, Yc, W, H float64
}

/*
ScaleBox init function

Args:

	x float64: Central x in ratio
	y float64: Central y in ratio
	w float64: Width in ratio
	h float64: Height in ratio

Returns:

	*ScaleBox: Pointer to a ScaleBox object
*/
func NewScl(x float64, y float64, w float64, h float64) *ScaleBox {
	return &ScaleBox{Xc: x, Yc: y, W: w, H: h}
}

type Box struct {
	ID   int
	ImSz Size
	Sz   Size
	Rct  Rect
	Scl  ScaleBox
}

/*
Size init function

Args:

	w int: Width
	h int: Height

Returns:

	*Size: Pointer to a Size object
*/
func NewSize(w int, h int) *Size {
	return &Size{W: w, H: h}
}

/*
Rect init function

Args:

	xtl float64: x value of top-left point
	ytl float64: y value of top-left point
	xbr float64: x value of bottom-right point
	ybr float64: y value of bottom-right point

Returns:

	*Rect: Pointer to a Rect object
*/
func NewRect(xtl int, ytl int, xbr int, ybr int) *Rect {
	return &Rect{Xtl: xtl, Ytl: ytl, Xbr: xbr, Ybr: ybr}
}

/*
Box init function, takes Rect in, then
calculate box size and scaled box:
[x_central, y_cenral, width, height]

Args:

	id int: Object class ID
	r *Rect: Rectangle represent a bounding box
	s *Sect: Image width and height

Returns:

	*Box: Pointer to a Box object
*/
func NewBox(id int, r *Rect, s *Size) *Box {
	box := Box{ID: id, ImSz: *s, Rct: *r}
	box.Sz.W = box.Rct.Xbr - box.Rct.Xtl + 1
	box.Sz.H = box.Rct.Ybr - box.Rct.Ytl + 1
	return &box
}

/*
Adjust box value taht out of boundary

Args:

	None

Returns:

	b *Box: Box object after trimed
*/
func (b *Box) Trim() *Box {
	if b.Rct.Xtl < 0 {
		b.Rct.Xtl = 0
	}
	if b.Rct.Ytl < 0 {
		b.Rct.Ytl = 0
	}
	if b.Rct.Xbr >= b.ImSz.W {
		b.Rct.Xbr = b.ImSz.W - 1
	}
	if b.Rct.Ybr >= b.ImSz.H {
		b.Rct.Ybr = b.ImSz.H - 1
	}
	return b
}

/*
Calcualte scaled box

Args:

	None

Returns:

	b *Box: Box object after scaled
*/
func (b *Box) Scale() *Box {
	b.Scl.Xc = (float64(b.Rct.Xtl) + float64(b.Rct.Xbr)) / 2 / float64(b.ImSz.W)
	b.Scl.Yc = (float64(b.Rct.Ytl) + float64(b.Rct.Ybr)) / 2 / float64(b.ImSz.H)
	b.Scl.W = float64(b.Sz.W) / float64(b.ImSz.W)
	b.Scl.H = float64(b.Sz.H) / float64(b.ImSz.H)
	return b
}

/*
Calcualte scaled box

Args:

	None

Returns:

	b *Box: Box object after unscaled
*/
func (b *Box) UnScale() *Box {
	b.Sz.W = int(b.Scl.W * float64(b.ImSz.W))
	b.Sz.H = int(b.Scl.H * float64(b.ImSz.H))
	b.Rct.Xtl = int((b.Scl.Xc*float64(b.ImSz.W)*2 - float64(b.Sz.W) + 1) / 2)
	b.Rct.Ytl = int((b.Scl.Yc*float64(b.ImSz.H)*2 - float64(b.Sz.H) + 1) / 2)
	b.Rct.Xbr = b.Rct.Xtl + b.Sz.W
	b.Rct.Ybr = b.Rct.Ytl + b.Sz.H
	return b
}
