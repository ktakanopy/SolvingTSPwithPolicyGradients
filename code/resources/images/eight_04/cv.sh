convert                                                  \
  -delay 10                                              \
   $(for i in $(seq 0 1 29); do echo rgb-${i}.jpg; done) \
  -loop 0                                                \
   animated.gif

