AR:=ar
CC:=gcc
SRCDIR:=src
DEMDIR:=demo
OBJDIR:=build
INCDIR:=include
BINDIR:=dist
LIBDIR:=$(BINDIR)

INCS:=$(wildcard $(SRCDIR)/*.h)
OBJS:=$(subst $(SRCDIR)/,$(OBJDIR)/,$(patsubst %.c,%.o,$(wildcard $(SRCDIR)/*.c)))

CFLAGS:=-std=gnu99 -Wall -pedantic -march=native -O2 -g
IFLAGS:=-I$(INCDIR)
# LFLAGS:=-L$(LIBDIR) -lgsl -lgslcblas -lgsgp -lm
LFLAGS:=-L$(LIBDIR) -lgsgp -lm

INC:=$(SRCDIR)/gsgp.h
LIB:=$(LIBDIR)/libgsgp.a
BIN:=$(BINDIR)/gsgp

all: $(LIB) $(BIN)

lib: $(LIB)

$(LIBDIR)/libgsgp.a: $(OBJS) $(INCS)
	@echo creating library $@ from $^
	@mkdir -p $(BINDIR)
	@$(AR) -r $@ $(OBJS)
	@echo copying headers to $(INCDIR)
	@mkdir -p $(INCDIR)
	@cp $(INC) $(INCDIR)

$(BINDIR)/gsgp: $(subst $(DEMDIR)/,$(OBJDIR)/,$(patsubst %.c,%.o,$(wildcard $(DEMDIR)/*.c))) $(LIB)
	@echo linking $@ from $^
	@$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(OBJDIR)/%.o : $(SRCDIR)/%.c $(INCS)
	@echo compiling $< into $@
	@mkdir -p $(OBJDIR)
	@$(CC) $(CFLAGS) $(IFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(DEMDIR)/%.c $(wildcard $(DEMDIR)/*.h) $(INC)
	@echo compiling $< into $@
	@$(CC) $(CFLAGS) $(IFLAGS) -c $< -o $@

clean:
	@rm -rf $(OBJDIR)

nuke: clean
	@rm -rf $(INCDIR) $(BINDIR) $(LIBDIR)

strip: all
	@echo running strip on $(BIN)
	@strip $(BIN)
