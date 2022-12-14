package tps

import (
	"os"
	"strconv"
	"strings"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
)

func SaveTXT(path string, id int, ob *Box, loc *Location, b *Box) {
	/*Calculate data and write to *.txt files.

	Args:
		cls([]string): classes you choose to generate
		path(string): TXT file path

	Returns:
		None
	*/

	str := []string{
		strconv.Itoa(id),
		op.F642Str(ob.Scl.Xc), op.F642Str(ob.Scl.Yc),
		op.F642Str(ob.Scl.W), op.F642Str(ob.Scl.H),
		op.F642Str(loc.Y), op.F642Str(loc.X),
		op.F642Str(b.Scl.Xc), op.F642Str(b.Scl.Yc),
		op.F642Str(b.Scl.W), op.F642Str(b.Scl.H),
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0755)
	op.CheckE(err)
	defer file.Close()
	file.WriteString(strings.Join(str, " "))
}
