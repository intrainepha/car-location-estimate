package types

import (
	"fmt"
	"math"
	"strings"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
)

type KITTI struct {
	Cls  string
	Trct float64
	Ocld int
	Rct  Rect
	Loc  Location
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

func (k *KITTI) Check(cls []string) (int, error) {
	/*Kitti init function

	Args:
		l(string): Line data from kitti label(*txt)

	Returns:
		(*KITTI): Pointer to KITTI object
	*/

	// Filter classes those not slected by user
	id, flag := k.getID(cls, k.Cls)
	if !flag {
		return -1, fmt.Errorf("class %s is not selected", k.Cls)
	}
	// info[1]: float from 0(non-truncated) to 1(truncated)
	if k.Trct != 0 {
		return -1, fmt.Errorf("sample is truncated")
	}
	// info[2]: 0->fully visible, 1->partial occluded, 2->largely occluded, 3=unknown
	if k.Ocld != 0 {
		return -1, fmt.Errorf("sample is occluded")
	}
	// x_range=[-8, 8], y_range=[0, 80], mutable
	if math.Abs(k.Loc.X) > 8 || k.Loc.Y < 0 || k.Loc.Y > 80 {
		return -1, fmt.Errorf("sample location is out of range")
	}

	return id, nil
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
