package ops

import (
	"log"
	"time"
)

/*
Print running time measurement

Args:

	start time.Time: start time
	name string: name for identification

Returns:

	None
*/
func Timer(start time.Time, name string) {
	elapsed := time.Since(start).Seconds()
	log.Printf("elapsed:%s:%f(s)", name, elapsed)
}
