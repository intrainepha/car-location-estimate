package tps

import (
	"os"
	"path"
	"strings"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
)

type File struct {
	Path    string
	File    os.File
	Content string
}

func NewFile(p string) (*File, error) {
	/*Open file, create a new one if it does not exist

	Args:
		p(string): file path

	Returns:
		(*File)
		(error)
	*/

	d := path.Dir(p)
	if !op.CheckDir(d) {
		err := os.MkdirAll(d, 0755)
		if err != nil {
			return &File{}, err
		}
	}
	f, err := os.OpenFile(p, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		return &File{}, err
	}

	return &File{Path: p, File: *f}, nil
}

func (t *File) Read() error {
	/*Load file content, create a new one if it does not exist

	Args:
		p(string): file path

	Returns:
		(*File)
		(error)
	*/

	bt, err := os.ReadFile(t.Path)
	if err != nil {
		return err
	}
	t.Content = strings.Trim(string(bt), "\n")

	return nil
}

func (t *File) ReadLines() []string {
	/*Read lines from *.File file.

	Args:
		None

	Returns:
		([]string): line data from file
	*/

	t.Read()

	return strings.Split(t.Content, "\n")
}

func (t *File) WriteLine(s string) error {
	/*Write line date to file buffer

	Args:
		None

	Returns:
		(error)
	*/

	info, err := os.Stat(t.Path)
	if err != nil {
		return err
	} else {
		if info.Size() == 0 {
			t.File.WriteString(s)
		} else {
			t.File.WriteString("\n" + s)
		}

	}

	return nil
}

func (t *File) Close() {
	/*Close os.File buffer.

	Args:
		None

	Returns:
		None
	*/

	t.File.Close()
}
