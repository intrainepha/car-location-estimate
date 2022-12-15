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

func (t *TXT) WriteLine(s string) {
	/*Write line date to file buffer

	Args:
		None

	Returns:
		([]string): Line data from text file
	*/

	t.File.WriteString(s + "\n")
}

func (t *TXT) Close() {
	/*Close os.File buffer.

	Args:
		None

	Returns:
		None
	*/

	t.File.Close()
}
