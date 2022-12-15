package tps

import (
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
	Cls  Cls
	Name string
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
		(*Location): pointer to a Location object
	*/

	return &Location{X: x, Y: y}
}

func NewOffset(x, y float64) *Offset {
	return &Offset{X: x, Y: y}
}

func NewKITTI(cls []string, l string) *KITTI {
	/*Kitti init function

	Args:
		l(string): line data from kitti label(*txt)

	Returns:
		(*KITTI): pointer to KITTI object
	*/

	info := strings.Split(l, " ")
	rct := *NewRect(
		op.Str2f64(info[4]), op.Str2f64(info[5]),
		op.Str2f64(info[6]), op.Str2f64(info[7]),
	)

	loc := *NewLoc(op.Str2f64(info[11]), op.Str2f64(info[13]))
	return &KITTI{
		Cls:  *NewCls(cls),
		Name: info[0],
		Trct: op.Str2f64(info[1]),
		Ocld: op.Str2int(info[2]),
		Rct:  rct,
		Loc:  loc,
	}
}

func (k *KITTI) Check(cls []string) bool {
	/*Kitti init function

	Args:
		l(string): line data from kitti label(*txt)

	Returns:
		(*KITTI): pointer to KITTI object
	*/

	if k.Cls.GetID(k.Name) == -1 {
		//Class is not selected
		return false
	}
	if k.Trct != 0 {
		// return fmt.Errorf("sample is truncated")
		return false
	}
	if k.Ocld != 0 {
		// return fmt.Errorf("sample is occluded")
		return false
	}
	if math.Abs(k.Loc.X) > 8 || k.Loc.Y < 0 || k.Loc.Y > 80 {
		// return fmt.Errorf("sample location is out of range")
		return false
	}

	return true
}

func (k *KITTI) MakeROI(imSz *Size, r *Rect, s float64) (*Box, *Box, *Box, *Offset) {
	/*Make ROI Box

	Args:
		id(int): class ID
		imSz(*Size): image width and height
		r(*Rect): rectangle
		s(float64): scale number

	Returns:
		ob(*Box): original Box relative to origin image
		rb(*Box): ROI Box relative to origin image
		b(*Box): object Box relative to ROI
	*/
	// TODO: trim and scale
	id := k.Cls.GetID(k.Name)
	ob := NewBox(id, r, imSz)
	off := NewOffset(ob.Sz.W*s, ob.Sz.H*s)
	rRct := NewRect(
		ob.Rct.Xtl-off.X, ob.Rct.Ytl-off.Y,
		ob.Rct.Xbr+off.X, ob.Rct.Ybr+off.Y,
	)
	rb := NewBox(id, rRct, imSz)
	oRct := NewRect(
		ob.Rct.Xtl-rb.Rct.Xtl, ob.Rct.Ytl-rb.Rct.Ytl,
		ob.Rct.Xbr-rb.Rct.Xtl, ob.Rct.Ybr-rb.Rct.Ytl,
	)
	b := NewBox(id, oRct, &rb.Sz)

	return ob, rb, b, off
}
