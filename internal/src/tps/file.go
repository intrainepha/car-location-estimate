package tps

import (
	"log"
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

/*
Open file, create a new one if it does not exist

Args:

	p string: file path

Returns:

	*File
*/
func NewFile(p string) *File {
	d := path.Dir(p)
	if !op.Exists(d) {
		err := os.MkdirAll(d, 0755)
		if err != nil {
			log.Panic(err)
		}
	}
	f, err := os.OpenFile(p, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		log.Panic(err)
	}
	return &File{Path: p, File: *f}
}

/*
Load file content, create a new one if it does not exist

Args:

	p string: file path

Returns:

	string: read content
*/
func (t *File) Read() string {
	bt, err := os.ReadFile(t.Path)
	if err != nil {
		log.Panic(err)
	}
	t.Content = strings.Trim(string(bt), "\n")

	return t.Content
}

/*
Read lines from *.File file.

Args:

	None

Returns:

	[]string: line data from file
*/
func (t *File) ReadLines() []string {
	t.Read()
	return strings.Split(t.Content, "\n")
}

/*
Write line date to file buffer

Args:

	None

Returns:

	None
*/
func (t *File) WriteLine(s string) {
	info, err := os.Stat(t.Path)
	if err != nil {
		log.Panic(err)
	} else {
		if info.Size() == 0 {
			t.File.WriteString(s)
		} else {
			t.File.WriteString("\n" + s)
		}

	}
}

/*
Close os.File buffer.

Args:

	None

Returns:

	None
*/
func (t *File) Close() {
	t.File.Close()
}
