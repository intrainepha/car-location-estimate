package tps

import (
	"os"
	"path"
	"strings"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
)

type TXT struct {
	Path    string
	File    os.File
	Content string
}

func NewTXT(p string) (*TXT, error) {
	/*Open *.txt file, create a new one if it does not exist

	Args:
		p(string): File path

	Returns:
		(*TXT)
		(error)
	*/

	d := path.Dir(p)
	if !op.CheckDir(d) {
		err := os.MkdirAll(d, 0755)
		if err != nil {
			return &TXT{}, err
		}
	}
	f, err := os.OpenFile(p, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		return &TXT{}, err
	}

	return &TXT{Path: p, File: *f}, nil
}

func (t *TXT) Load() error {
	/*Load *.txt file, create a new one if it does not exist

	Args:
		p(string): File path

	Returns:
		(*TXT)
		(error)
	*/

	bt, err := os.ReadFile(t.Path)
	if err != nil {
		return err
	}
	t.Content = strings.Trim(string(bt), "\n")

	return nil
}

func (t *TXT) ReadLines() []string {
	/*Load *.txt file.

	Args:
		None

	Returns:
		([]string): Line data from text file
	*/

	return strings.Split(t.Content, "\n")
}

func (t *TXT) WriteLine(ss []string, sep string) {
	/*Write line date to file buffer

	Args:
		None

	Returns:
		([]string): Line data from text file
	*/

	t.File.WriteString(strings.Join(ss, sep))
}

func (t *TXT) Save(p string) {
	/*Calculate data and write to *.txt files.

	Args:
		cls([]string): classes you choose to generate
		path(string): TXT file path

	Returns:
		None
	*/

	// str := []string{
	// 	strconv.Itoa(id),
	// 	op.F642Str(ob.Scl.Xc), op.F642Str(ob.Scl.Yc),
	// 	op.F642Str(ob.Scl.W), op.F642Str(ob.Scl.H),
	// 	op.F642Str(loc.Y), op.F642Str(loc.X),
	// 	op.F642Str(b.Scl.Xc), op.F642Str(b.Scl.Yc),
	// 	op.F642Str(b.Scl.W), op.F642Str(b.Scl.H),
	// }
	t.File.WriteString(strings.Join(str, " "))
}
