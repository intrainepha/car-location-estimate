package tps

import (
	"fmt"
	"math"
	"strings"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
)

type Location struct {
	X float64
	Y float64
}

type Offset struct {
	X float64
	Y float64
}

type KITTI struct {
	Cls  string
	Trct float64
	Ocld int
	Rct  Rect
	Loc  Location
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

func NewOffset(x, y float64) *Offset {
	return &Offset{X: x, Y: y}
}

func NewKITTI(l string) *KITTI {
	/*Kitti init function

	Args:
		l(string): Line data from kitti label(*txt)

	Returns:
		(*KITTI): Pointer to KITTI object
	*/

	info := strings.Split(l, " ")
	rct := *NewRect(
		op.Str2f64(info[4]), op.Str2f64(info[5]),
		op.Str2f64(info[6]), op.Str2f64(info[7]),
	)

	loc := *NewLoc(op.Str2f64(info[11]), op.Str2f64(info[13]))
	return &KITTI{
		Cls:  info[0],
		Trct: op.Str2f64(info[1]),
		Ocld: op.Str2int(info[2]),
		Rct:  rct,
		Loc:  loc,
	}
}

func (k *KITTI) getID(ss []string, s string) (int, bool) {
	/*Check if a string exists in a slice of string

	Args:
		str(string): string data

	Returns:
		(bool): Check result
	*/

	for i, v := range ss {
		if v == s {
			return i, true
		}
	}

	return -1, false
}

func (k *KITTI) Check(cls []string) (int, error) {
	/*Kitti init function

	Args:
		l(string): Line data from kitti label(*txt)

	Returns:
		(*KITTI): Pointer to KITTI object
	*/

	id, flag := k.getID(cls, k.Cls)
	if !flag {
		return -1, fmt.Errorf("class %s is not selected", k.Cls)
	}
	if k.Trct != 0 {
		return -1, fmt.Errorf("sample is truncated")
	}
	if k.Ocld != 0 {
		return -1, fmt.Errorf("sample is occluded")
	}
	if math.Abs(k.Loc.X) > 8 || k.Loc.Y < 0 || k.Loc.Y > 80 {
		return -1, fmt.Errorf("sample location is out of range")
	}

	return id, nil
}

func (k *KITTI) MakeROI(imSz *Size, r *Rect, s float64) (*Box, *Box, *Box, *Offset) {
	/*Make ROI Box

	Args:
		imSz(*Size): Image width and height
		r(*Rect): Rectangle
		s(float64): Scale number

	Returns:
		ob(*Box): Original Box relative to origin image
		rb(*Box): ROI Box relative to origin image
		b(*Box): Object Box relative to ROI
	*/

	ob := NewBox(r, imSz)
	off := NewOffset(ob.Sz.W*s, ob.Sz.H*s)
	rRct := NewRect(
		ob.Rct.Xtl-off.X, ob.Rct.Ytl-off.Y,
		ob.Rct.Xbr+off.X, ob.Rct.Ybr+off.Y,
	)
	rb := NewBox(rRct, imSz)
	oRct := NewRect(
		ob.Rct.Xtl-rb.Rct.Xtl, ob.Rct.Ytl-rb.Rct.Ytl,
		ob.Rct.Xbr-rb.Rct.Xtl, ob.Rct.Ybr-rb.Rct.Ytl,
	)
	b := NewBox(oRct, &rb.Sz)

	return ob, rb, b, off
}
