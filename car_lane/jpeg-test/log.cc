#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include<sys/time.h>
#include<unistd.h>
#include "log.h"




static LOG_LEVEL local_level = LOG_VERBOSE_LEVEL;
void write_log_imp(const char *tag, const char *format, ...)
{
    #define  BUFFER_SIZE               2048
    char buf[BUFFER_SIZE + 1];
    va_list arg;
    va_start(arg, format);
    vsnprintf(buf, BUFFER_SIZE, format, arg);
    va_end(arg);
    #undef BUFFER_SIZE
    puts(buf);
}

 void set_log_level(LOG_LEVEL log_level)
{
  local_level = log_level;
}

LOG_LEVEL get_log_level(void) {
  return local_level;
}

unsigned long long log_get_timestamp(void)
{
  struct  timeval   tv;
  gettimeofday(&tv, NULL);  
  return tv.tv_sec * 1000000 + tv.tv_usec ;
}

FUNC_WRITE_LOP_IMP* pfunc_write_log_imp = write_log_imp;
