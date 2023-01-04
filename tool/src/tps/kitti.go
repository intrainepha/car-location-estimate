package tps

import (
	"math"
	"strings"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
	"golang.org/x/exp/constraints"
)

type number interface {
	constraints.Integer | constraints.Float
}

type Point[T number] struct {
	X, Y T
}

/*
Point initializtion

Args:

	x int|float64: x value of a point
	y int|float64: y value of a point

Returns:

	*Point: pointer to a Point object
*/
func NewPoint[T number](x, y T) *Point[T] {
	return &Point[T]{X: x, Y: y}
}

type KITTI struct {
	Cls  Cls
	Name string
	Trct float64
	Ocld int
	Rct  Rect
	Loc  Point[float64]
}

/*
Kitti initialization

Args:

	l string: line data from kitti label(*txt)

Returns:

	*KITTI: pointer to KITTI object
*/
func NewKITTI(cls []string, l string) *KITTI {
	info := strings.Split(l, " ")
	rct := *NewRect(
		int(op.Stof(info[4])), int(op.Stof(info[5])),
		int(op.Stof(info[6])), int(op.Stof(info[7])),
	)

	loc := *NewPoint(op.Stof(info[11]), op.Stof(info[13]))
	return &KITTI{
		Cls:  *NewCls(cls),
		Name: info[0],
		Trct: op.Stof(info[1]),
		Ocld: op.Stoi(info[2]),
		Rct:  rct,
		Loc:  loc,
	}
}

/*
Filter with conditions:
 1. Class is not selected
 2. Sample is truncated
 3. Sample is occluded
 4. Sample location is out of range

Args:

	l string: line data from kitti label(*txt)

Returns:

	bool: filtering status
*/
func (k *KITTI) FilterOut() bool {
	if k.Cls.GetID(k.Name) == -1 {
		return true
	}
	if k.Trct != 0 {
		return true
	}
	if k.Ocld != 0 {
		return true
	}
	if math.Abs(k.Loc.X) > 8 || k.Loc.Y < 0 || k.Loc.Y > 80 {
		return true
	}
	return false
}

/*
Make ROI Box

Args:

	id int: class ID
	imsz *Size: image width and height
	r *Rect: rectangle
	t [4]float64: translation factors
	s float64: scale factors

Returns:

	ob *Box: original Box relative to origin image
	rb *Box: ROI Box relative to origin image
	b *Box: object Box relative to ROI
	off *Point: offset to ROI box
*/
func (k *KITTI) MakeROI(imsz *Size, r *Rect, t [4]float64, s float64) (*Box, *Box, *Box, *Point[int]) {
	tRct := NewRect(
		int(float64(r.Xtl)+t[0]), int(float64(r.Ytl)+t[1]),
		int(float64(r.Xbr)+t[2]), int(float64(r.Ybr)+t[3]),
	)
	id := k.Cls.GetID(k.Name)
	ob := NewBox(id, tRct, imsz)
	off := NewPoint(int(float64(ob.Sz.W)*s), int(float64(ob.Sz.H)*s))
	rRct := NewRect(
		ob.Rct.Xtl-off.X, ob.Rct.Ytl-off.Y,
		ob.Rct.Xbr+off.X, ob.Rct.Ybr+off.Y,
	)
	rb := NewBox(id, rRct, imsz).Trim()
	rct := NewRect(
		r.Xtl-rb.Rct.Xtl, r.Ytl-rb.Rct.Ytl,
		r.Xbr-rb.Rct.Xtl, r.Ybr-rb.Rct.Ytl,
	)
	b := NewBox(id, rct, &rb.Sz).Trim().Scale()
	return ob, rb, b, off
}
