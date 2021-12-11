import weakref
from os.path import abspath
from os.path import splitext
import sys
import warnings
import numpy as np

cimport cython
cimport numpy as np
from scipy.stats import gmean, iqr
from cpython cimport Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free
from libc.stdio cimport fdopen
from numpy.math cimport INFINITY, NAN


include "expr.pxi"
include "lp.pxi"
include "branchrule.pxi"
include "conshdlr.pxi"
include "event.pxi"
include "heuristic.pxi"
include "presol.pxi"
include "pricer.pxi"
include "propagator.pxi"
include "sepa.pxi"

# recommended SCIP version; major version is required
MAJOR = 5
MINOR = 0
PATCH = 1

# for external user functions use def; for functions used only inside the interface (starting with _) use cdef
# todo: check whether this is currently done like this

if sys.version_info >= (3, 0):
    str_conversion = lambda x:bytes(x,'utf-8')
else:
    str_conversion = lambda x:x

from libc.stdio cimport printf

# Mapping the SCIP_RESULT enum to a python class
# This is required to return SCIP_RESULT in the python code
# In __init__.py this is imported as SCIP_RESULT to keep the
# original naming scheme using capital letters
cdef class PY_SCIP_RESULT:
    DIDNOTRUN   = SCIP_DIDNOTRUN
    DELAYED     = SCIP_DELAYED
    DIDNOTFIND  = SCIP_DIDNOTFIND
    FEASIBLE    = SCIP_FEASIBLE
    INFEASIBLE  = SCIP_INFEASIBLE
    UNBOUNDED   = SCIP_UNBOUNDED
    CUTOFF      = SCIP_CUTOFF
    SEPARATED   = SCIP_SEPARATED
    NEWROUND    = SCIP_NEWROUND
    REDUCEDDOM  = SCIP_REDUCEDDOM
    CONSADDED   = SCIP_CONSADDED
    CONSCHANGED = SCIP_CONSCHANGED
    BRANCHED    = SCIP_BRANCHED
    SOLVELP     = SCIP_SOLVELP
    FOUNDSOL    = SCIP_FOUNDSOL
    SUSPENDED   = SCIP_SUSPENDED
    SUCCESS     = SCIP_SUCCESS

cdef class PY_SCIP_PARAMSETTING:
    DEFAULT     = SCIP_PARAMSETTING_DEFAULT
    AGGRESSIVE  = SCIP_PARAMSETTING_AGGRESSIVE
    FAST        = SCIP_PARAMSETTING_FAST
    OFF         = SCIP_PARAMSETTING_OFF

cdef class PY_SCIP_PARAMEMPHASIS:
    DEFAULT      = SCIP_PARAMEMPHASIS_DEFAULT
    CPSOLVER     = SCIP_PARAMEMPHASIS_CPSOLVER
    EASYCIP      = SCIP_PARAMEMPHASIS_EASYCIP
    FEASIBILITY  = SCIP_PARAMEMPHASIS_FEASIBILITY
    HARDLP       = SCIP_PARAMEMPHASIS_HARDLP
    OPTIMALITY   = SCIP_PARAMEMPHASIS_OPTIMALITY
    COUNTER      = SCIP_PARAMEMPHASIS_COUNTER
    PHASEFEAS    = SCIP_PARAMEMPHASIS_PHASEFEAS
    PHASEIMPROVE = SCIP_PARAMEMPHASIS_PHASEIMPROVE
    PHASEPROOF   = SCIP_PARAMEMPHASIS_PHASEPROOF

cdef class PY_SCIP_STATUS:
    UNKNOWN        = SCIP_STATUS_UNKNOWN
    USERINTERRUPT  = SCIP_STATUS_USERINTERRUPT
    NODELIMIT      = SCIP_STATUS_NODELIMIT
    TOTALNODELIMIT = SCIP_STATUS_TOTALNODELIMIT
    STALLNODELIMIT = SCIP_STATUS_STALLNODELIMIT
    TIMELIMIT      = SCIP_STATUS_TIMELIMIT
    MEMLIMIT       = SCIP_STATUS_MEMLIMIT
    GAPLIMIT       = SCIP_STATUS_GAPLIMIT
    SOLLIMIT       = SCIP_STATUS_SOLLIMIT
    BESTSOLLIMIT   = SCIP_STATUS_BESTSOLLIMIT
    RESTARTLIMIT   = SCIP_STATUS_RESTARTLIMIT
    OPTIMAL        = SCIP_STATUS_OPTIMAL
    INFEASIBLE     = SCIP_STATUS_INFEASIBLE
    UNBOUNDED      = SCIP_STATUS_UNBOUNDED
    INFORUNBD      = SCIP_STATUS_INFORUNBD

cdef class PY_SCIP_STAGE:
    INIT         = SCIP_STAGE_INIT
    PROBLEM      = SCIP_STAGE_PROBLEM
    TRANSFORMING = SCIP_STAGE_TRANSFORMING
    TRANSFORMED  = SCIP_STAGE_TRANSFORMED
    INITPRESOLVE = SCIP_STAGE_INITPRESOLVE
    PRESOLVING   = SCIP_STAGE_PRESOLVING
    EXITPRESOLVE = SCIP_STAGE_EXITPRESOLVE
    PRESOLVED    = SCIP_STAGE_PRESOLVED
    INITSOLVE    = SCIP_STAGE_INITSOLVE
    SOLVING      = SCIP_STAGE_SOLVING
    SOLVED       = SCIP_STAGE_SOLVED
    EXITSOLVE    = SCIP_STAGE_EXITSOLVE
    FREETRANS    = SCIP_STAGE_FREETRANS
    FREE         = SCIP_STAGE_FREE

cdef class PY_SCIP_NODETYPE:
    FOCUSNODE   = SCIP_NODETYPE_FOCUSNODE
    PROBINGNODE = SCIP_NODETYPE_PROBINGNODE
    SIBLING     = SCIP_NODETYPE_SIBLING
    CHILD       = SCIP_NODETYPE_CHILD
    LEAF        = SCIP_NODETYPE_LEAF
    DEADEND     = SCIP_NODETYPE_DEADEND
    JUNCTION    = SCIP_NODETYPE_JUNCTION
    PSEUDOFORK  = SCIP_NODETYPE_PSEUDOFORK
    FORK        = SCIP_NODETYPE_FORK
    SUBROOT     = SCIP_NODETYPE_SUBROOT
    REFOCUSNODE = SCIP_NODETYPE_REFOCUSNODE


cdef class PY_SCIP_PROPTIMING:
    BEFORELP     = SCIP_PROPTIMING_BEFORELP
    DURINGLPLOOP = SCIP_PROPTIMING_DURINGLPLOOP
    AFTERLPLOOP  = SCIP_PROPTIMING_AFTERLPLOOP
    AFTERLPNODE  = SCIP_PROPTIMING_AFTERLPNODE

cdef class PY_SCIP_PRESOLTIMING:
    NONE       = SCIP_PRESOLTIMING_NONE
    FAST       = SCIP_PRESOLTIMING_FAST
    MEDIUM     = SCIP_PRESOLTIMING_MEDIUM
    EXHAUSTIVE = SCIP_PRESOLTIMING_EXHAUSTIVE

cdef class PY_SCIP_HEURTIMING:
    BEFORENODE        = SCIP_HEURTIMING_BEFORENODE
    DURINGLPLOOP      = SCIP_HEURTIMING_DURINGLPLOOP
    AFTERLPLOOP       = SCIP_HEURTIMING_AFTERLPLOOP
    AFTERLPNODE       = SCIP_HEURTIMING_AFTERLPNODE
    AFTERPSEUDONODE   = SCIP_HEURTIMING_AFTERPSEUDONODE
    AFTERLPPLUNGE     = SCIP_HEURTIMING_AFTERLPPLUNGE
    AFTERPSEUDOPLUNGE = SCIP_HEURTIMING_AFTERPSEUDOPLUNGE
    DURINGPRICINGLOOP = SCIP_HEURTIMING_DURINGPRICINGLOOP
    BEFOREPRESOL      = SCIP_HEURTIMING_BEFOREPRESOL
    DURINGPRESOLLOOP  = SCIP_HEURTIMING_DURINGPRESOLLOOP
    AFTERPROPLOOP     = SCIP_HEURTIMING_AFTERPROPLOOP

cdef class PY_SCIP_EVENTTYPE:
    DISABLED        = SCIP_EVENTTYPE_DISABLED
    VARADDED        = SCIP_EVENTTYPE_VARADDED
    VARDELETED      = SCIP_EVENTTYPE_VARDELETED
    VARFIXED        = SCIP_EVENTTYPE_VARFIXED
    VARUNLOCKED     = SCIP_EVENTTYPE_VARUNLOCKED
    OBJCHANGED      = SCIP_EVENTTYPE_OBJCHANGED
    GLBCHANGED      = SCIP_EVENTTYPE_GLBCHANGED
    GUBCHANGED      = SCIP_EVENTTYPE_GUBCHANGED
    LBTIGHTENED     = SCIP_EVENTTYPE_LBTIGHTENED
    LBRELAXED       = SCIP_EVENTTYPE_LBRELAXED
    UBTIGHTENED     = SCIP_EVENTTYPE_UBTIGHTENED
    UBRELAXED       = SCIP_EVENTTYPE_UBRELAXED
    GHOLEADDED      = SCIP_EVENTTYPE_GHOLEADDED
    GHOLEREMOVED    = SCIP_EVENTTYPE_GHOLEREMOVED
    LHOLEADDED      = SCIP_EVENTTYPE_LHOLEADDED
    LHOLEREMOVED    = SCIP_EVENTTYPE_LHOLEREMOVED
    IMPLADDED       = SCIP_EVENTTYPE_IMPLADDED
    PRESOLVEROUND   = SCIP_EVENTTYPE_PRESOLVEROUND
    NODEFOCUSED     = SCIP_EVENTTYPE_NODEFOCUSED
    NODEFEASIBLE    = SCIP_EVENTTYPE_NODEFEASIBLE
    NODEINFEASIBLE  = SCIP_EVENTTYPE_NODEINFEASIBLE
    NODEBRANCHED    = SCIP_EVENTTYPE_NODEBRANCHED
    FIRSTLPSOLVED   = SCIP_EVENTTYPE_FIRSTLPSOLVED
    LPSOLVED        = SCIP_EVENTTYPE_LPSOLVED
    LPEVENT         = SCIP_EVENTTYPE_LPEVENT
    POORSOLFOUND    = SCIP_EVENTTYPE_POORSOLFOUND
    BESTSOLFOUND    = SCIP_EVENTTYPE_BESTSOLFOUND
    ROWADDEDSEPA    = SCIP_EVENTTYPE_ROWADDEDSEPA
    ROWDELETEDSEPA  = SCIP_EVENTTYPE_ROWDELETEDSEPA
    ROWADDEDLP      = SCIP_EVENTTYPE_ROWADDEDLP
    ROWDELETEDLP    = SCIP_EVENTTYPE_ROWDELETEDLP
    ROWCOEFCHANGED  = SCIP_EVENTTYPE_ROWCOEFCHANGED
    ROWCONSTCHANGED = SCIP_EVENTTYPE_ROWCONSTCHANGED
    ROWSIDECHANGED  = SCIP_EVENTTYPE_ROWSIDECHANGED
    SYNC            = SCIP_EVENTTYPE_SYNC

    NODESOLVED     = SCIP_EVENTTYPE_NODESOLVED


def PY_SCIP_CALL(SCIP_RETCODE rc):
    if rc == SCIP_OKAY:
        pass
    elif rc == SCIP_ERROR:
        raise Exception('SCIP: unspecified error!')
    elif rc == SCIP_NOMEMORY:
        raise MemoryError('SCIP: insufficient memory error!')
    elif rc == SCIP_READERROR:
        raise IOError('SCIP: read error!')
    elif rc == SCIP_WRITEERROR:
        raise IOError('SCIP: write error!')
    elif rc == SCIP_NOFILE:
        raise IOError('SCIP: file not found error!')
    elif rc == SCIP_FILECREATEERROR:
        raise IOError('SCIP: cannot create file!')
    elif rc == SCIP_LPERROR:
        raise Exception('SCIP: error in LP solver!')
    elif rc == SCIP_NOPROBLEM:
        raise Exception('SCIP: no problem exists!')
    elif rc == SCIP_INVALIDCALL:
        raise Exception('SCIP: method cannot be called at this time'
                            + ' in solution process!')
    elif rc == SCIP_INVALIDDATA:
        raise Exception('SCIP: error in input data!')
    elif rc == SCIP_INVALIDRESULT:
        raise Exception('SCIP: method returned an invalid result code!')
    elif rc == SCIP_PLUGINNOTFOUND:
        raise Exception('SCIP: a required plugin was not found !')
    elif rc == SCIP_PARAMETERUNKNOWN:
        raise KeyError('SCIP: the parameter with the given name was not found!')
    elif rc == SCIP_PARAMETERWRONGTYPE:
        raise LookupError('SCIP: the parameter is not of the expected type!')
    elif rc == SCIP_PARAMETERWRONGVAL:
        raise ValueError('SCIP: the value is invalid for the given parameter!')
    elif rc == SCIP_KEYALREADYEXISTING:
        raise KeyError('SCIP: the given key is already existing in table!')
    elif rc == SCIP_MAXDEPTHLEVEL:
        raise Exception('SCIP: maximal branching depth level exceeded!')
    else:
        raise Exception('SCIP: unknown return code!')

cdef class Event:
    cdef SCIP_EVENT* event

    @staticmethod
    cdef create(SCIP_EVENT* scip_event):
        event = Event()
        event.event = scip_event
        return event

    def getType(self):
        return SCIPeventGetType(self.event)

    def __repr__(self):
        return self.getType()

cdef class Column:
    """Base class holding a pointer to corresponding SCIP_COL"""
    cdef SCIP_COL* col

    @staticmethod
    cdef create(SCIP_COL* scip_col):
        col = Column()
        col.col = scip_col
        return col

    def getLPPos(self):
        return SCIPcolGetLPPos(self.col)

    def getBasisStatus(self):
        """Note: returns basis status `zero` for columns not in the current SCIP LP"""
        cdef SCIP_BASESTAT stat = SCIPcolGetBasisStatus(self.col)
        if stat == SCIP_BASESTAT_LOWER:
            return "lower"
        elif stat == SCIP_BASESTAT_BASIC:
            return "basic"
        elif stat == SCIP_BASESTAT_UPPER:
            return "upper"
        elif stat == SCIP_BASESTAT_ZERO:
            return "zero"
        else:
            raise Exception('SCIP returned unknown base status!')

    def isIntegral(self):
        return SCIPcolIsIntegral(self.col)

    def getVar(self):
        """gets variable this column represents"""
        cdef SCIP_VAR* var = SCIPcolGetVar(self.col)
        return Variable.create(var)

    def getPrimsol(self):
        """gets the primal LP solution of a column"""
        return SCIPcolGetPrimsol(self.col)

    def getLb(self):
        return SCIPcolGetLb(self.col)

    def getUb(self):
        return SCIPcolGetUb(self.col)

cdef class Row:
    """Base class holding a pointer to corresponding SCIP_ROW"""
    cdef SCIP_ROW* row

    @staticmethod
    cdef create(SCIP_ROW* scip_row):
        row = Row()
        row.row = scip_row
        return row

    def getLhs(self):
        return SCIProwGetLhs(self.row)

    def getRhs(self):
        return SCIProwGetRhs(self.row)

    def getConstant(self):
        return SCIProwGetConstant(self.row)

    def getLPPos(self):
        return SCIProwGetLPPos(self.row)

    def getBasisStatus(self):
        """Note: returns basis status `basic` for rows not in the current SCIP LP"""
        cdef SCIP_BASESTAT stat = SCIProwGetBasisStatus(self.row)
        if stat == SCIP_BASESTAT_LOWER:
            return "lower"
        elif stat == SCIP_BASESTAT_BASIC:
            return "basic"
        elif stat == SCIP_BASESTAT_UPPER:
            return "upper"
        elif stat == SCIP_BASESTAT_ZERO:
            # this shouldn't happen!
            raise Exception('SCIP returned base status zero for a row!')
        else:
            raise Exception('SCIP returned unknown base status!')

    def isIntegral(self):
        return SCIProwIsIntegral(self.row)

    def isModifiable(self):
        return SCIProwIsModifiable(self.row)

    def getNNonz(self):
        """get number of nonzero entries in row vector"""
        return SCIProwGetNNonz(self.row)

    def getNLPNonz(self):
        """get number of nonzero entries in row vector that correspond to columns currently in the SCIP LP"""
        return SCIProwGetNLPNonz(self.row)

    def getCols(self):
        """gets list with columns of nonzero entries"""
        cdef SCIP_COL** cols = SCIProwGetCols(self.row)
        return [Column.create(cols[i]) for i in range(self.getNNonz())]

    def getVals(self):
        """gets list with coefficients of nonzero entries"""
        cdef SCIP_Real* vals = SCIProwGetVals(self.row)
        return [vals[i] for i in range(self.getNNonz())]

cdef class Solution:
    """Base class holding a pointer to corresponding SCIP_SOL"""
    cdef SCIP_SOL* sol

    @staticmethod
    cdef create(SCIP_SOL* scip_sol):
        sol = Solution()
        sol.sol = scip_sol
        return sol

cdef class Node:
    """Base class holding a pointer to corresponding SCIP_NODE"""
    cdef SCIP_NODE* node

    @staticmethod
    cdef create(SCIP_NODE* scip_node):
        node = Node()
        node.node = scip_node
        return node

    def getParent(self):
        """Retrieve parent node."""
        return Node.create(SCIPnodeGetParent(self.node))

    def getNumber(self):
        """Retrieve number of node."""
        return SCIPnodeGetNumber(self.node)

    def getDepth(self):
        """Retrieve depth of node."""
        return SCIPnodeGetDepth(self.node)

    def getType(self):
        """Retrieve type of node."""
        return SCIPnodeGetType(self.node)

    def getLowerbound(self):
        """Retrieve lower bound of node."""
        return SCIPnodeGetLowerbound(self.node)

    def getEstimate(self):
        """Retrieve the estimated value of the best feasible solution in subtree of the node"""
        return SCIPnodeGetEstimate(self.node)

    def getNAddedConss(self):
        """Retrieve number of added constraints at this node"""
        return SCIPnodeGetNAddedConss(self.node)

    def isActive(self):
        """Is the node in the path to the current node?"""
        return SCIPnodeIsActive(self.node)

    def isPropagatedAgain(self):
        """Is the node marked to be propagated again?"""
        return SCIPnodeIsPropagatedAgain(self.node)

    def getBranchInfos(self):
        """Get branching decision of the parent node."""
        domchg = SCIPnodeGetDomchg(self.node)
        nboundchgs = SCIPdomchgGetNBoundchgs(domchg)
        assert nboundchgs == 1
        boundchg = SCIPdomchgGetBoundchg(domchg, 0)

        result = []
        result.append(SCIPboundchgGetNewbound(boundchg))
        result.append(Variable.create(SCIPboundchgGetVar(boundchg)))
        result.append(SCIPboundchgGetBoundchgtype(boundchg))
        result.append(SCIPboundchgGetBoundtype(boundchg))
        result.append(SCIPboundchgIsRedundant(boundchg))
        return result


cdef class Variable(Expr):
    """Is a linear expression and has SCIP_VAR*"""
    cdef SCIP_VAR* var

    @staticmethod
    cdef create(SCIP_VAR* scipvar):
        var = Variable()
        var.var = scipvar
        Expr.__init__(var, {Term(var) : 1.0})
        return var

    property name:
        def __get__(self):
            cname = bytes( SCIPvarGetName(self.var) )
            return cname.decode('utf-8')

    def ptr(self):
        """ """
        return <size_t>(self.var)

    def __repr__(self):
        return self.name

    def vtype(self):
        """Retrieve the variables type (BINARY, INTEGER or CONTINUOUS)"""
        vartype = SCIPvarGetType(self.var)
        if vartype == SCIP_VARTYPE_BINARY:
            return "BINARY"
        elif vartype == SCIP_VARTYPE_INTEGER:
            return "INTEGER"
        elif vartype == SCIP_VARTYPE_CONTINUOUS or vartype == SCIP_VARTYPE_IMPLINT:
            return "CONTINUOUS"

    def isOriginal(self):
        """Retrieve whether the variable belongs to the original problem"""
        return SCIPvarIsOriginal(self.var)

    def isInLP(self):
        """Retrieve whether the variable is a COLUMN variable that is member of the current LP"""
        return SCIPvarIsInLP(self.var)

    def getIndex(self):
        return SCIPvarGetIndex(self.var)

    def getCol(self):
        """Retrieve column of COLUMN variable"""
        cdef SCIP_COL* scip_col
        scip_col = SCIPvarGetCol(self.var)
        return Column.create(scip_col)

    def getLbOriginal(self):
        """Retrieve original lower bound of variable"""
        return SCIPvarGetLbOriginal(self.var)

    def getUbOriginal(self):
        """Retrieve original upper bound of variable"""
        return SCIPvarGetUbOriginal(self.var)

    def getLbGlobal(self):
        """Retrieve global lower bound of variable"""
        return SCIPvarGetLbGlobal(self.var)

    def getUbGlobal(self):
        """Retrieve global upper bound of variable"""
        return SCIPvarGetUbGlobal(self.var)

    def getLbLocal(self):
        """Retrieve current lower bound of variable"""
        return SCIPvarGetLbLocal(self.var)

    def getUbLocal(self):
        """Retrieve current upper bound of variable"""
        return SCIPvarGetUbLocal(self.var)

    def getObj(self):
        """Retrieve current objective value of variable"""
        return SCIPvarGetObj(self.var)

    def getLPSol(self):
        """Retrieve the current LP solution value of variable"""
        return SCIPvarGetLPSol(self.var)


cdef class Constraint:
    cdef SCIP_CONS* cons
    cdef public object data #storage for python user

    @staticmethod
    cdef create(SCIP_CONS* scipcons):
        if scipcons == NULL:
            raise Warning("cannot create Constraint with SCIP_CONS* == NULL")
        cons = Constraint()
        cons.cons = scipcons
        return cons

    property name:
        def __get__(self):
            cname = bytes( SCIPconsGetName(self.cons) )
            return cname.decode('utf-8')

    def __repr__(self):
        return self.name

    def isOriginal(self):
        """Retrieve whether the constraint belongs to the original problem"""
        return SCIPconsIsOriginal(self.cons)

    def isInitial(self):
        """Retrieve True if the relaxation of the constraint should be in the initial LP"""
        return SCIPconsIsInitial(self.cons)

    def isSeparated(self):
        """Retrieve True if constraint should be separated during LP processing"""
        return SCIPconsIsSeparated(self.cons)

    def isEnforced(self):
        """Retrieve True if constraint should be enforced during node processing"""
        return SCIPconsIsEnforced(self.cons)

    def isChecked(self):
        """Retrieve True if constraint should be checked for feasibility"""
        return SCIPconsIsChecked(self.cons)

    def isPropagated(self):
        """Retrieve True if constraint should be propagated during node processing"""
        return SCIPconsIsPropagated(self.cons)

    def isLocal(self):
        """Retrieve True if constraint is only locally valid or not added to any (sub)problem"""
        return SCIPconsIsLocal(self.cons)

    def isModifiable(self):
        """Retrieve True if constraint is modifiable (subject to column generation)"""
        return SCIPconsIsModifiable(self.cons)

    def isDynamic(self):
        """Retrieve True if constraint is subject to aging"""
        return SCIPconsIsDynamic(self.cons)

    def isRemovable(self):
        """Retrieve True if constraint's relaxation should be removed from the LP due to aging or cleanup"""
        return SCIPconsIsRemovable(self.cons)

    def isStickingAtNode(self):
        """Retrieve True if constraint is only locally valid or not added to any (sub)problem"""
        return SCIPconsIsStickingAtNode(self.cons)

    def isLinear(self):
        """Retrieve True if constraint is linear"""
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.cons))).decode('UTF-8')
        return constype == 'linear'

    def isQuadratic(self):
        """Retrieve True if constraint is quadratic"""
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(self.cons))).decode('UTF-8')
        return constype == 'quadratic'

# - remove create(), includeDefaultPlugins(), createProbBasic() methods
# - replace free() by "destructor"
# - interface SCIPfreeProb()
cdef class Model:
    cdef SCIP* _scip
    # store best solution to get the solution values easier
    cdef Solution _bestSol
    # can be used to store problem data
    cdef public object data
    # make Model weak referentiable
    cdef object __weakref__

    def __init__(self, problemName='model', defaultPlugins=True):
        """
        :param problemName: name of the problem (default 'model')
        :param defaultPlugins: use default plugins? (default True)
        """
        if self.version() < MAJOR:
            raise Exception("linked SCIP is not compatible to this version of PySCIPOpt - use at least version", MAJOR)
        if self.version() < MAJOR + MINOR/10.0 + PATCH/100.0:
            warnings.warn("linked SCIP {} is not recommended for this version of PySCIPOpt - use version {}.{}.{}".format(self.version(), MAJOR, MINOR, PATCH))
        self.create()
        self._bestSol = None
        if defaultPlugins:
            self.includeDefaultPlugins()
        self.createProbBasic(problemName)

    def __dealloc__(self):
        # call C function directly, because we can no longer call this object's methods, according to
        # http://docs.cython.org/src/reference/extension_types.html#finalization-dealloc
        PY_SCIP_CALL( SCIPfree(&self._scip) )

    def create(self):
        """Create a new SCIP instance"""
        PY_SCIP_CALL(SCIPcreate(&self._scip))

    def includeDefaultPlugins(self):
        """Includes all default plug-ins into SCIP"""
        PY_SCIP_CALL(SCIPincludeDefaultPlugins(self._scip))

    def createProbBasic(self, problemName='model'):
        """Create new problem instance with given name

        :param problemName: name of model or problem (Default value = 'model')

        """
        n = str_conversion(problemName)
        PY_SCIP_CALL(SCIPcreateProbBasic(self._scip, n))

    def freeProb(self):
        """Frees problem and solution process data"""
        PY_SCIP_CALL(SCIPfreeProb(self._scip))

    def freeTransform(self):
        """Frees all solution process data including presolving and transformed problem, only original problem is kept"""
        PY_SCIP_CALL(SCIPfreeTransform(self._scip))

    def version(self):
        """Retrieve SCIP version"""
        return SCIPversion()

    def printVersion(self):
        """Print version, copyright information and compile mode"""
        SCIPprintVersion(self._scip, NULL)

    def getProbName(self):
        """Retrieve problem name"""
        return bytes(SCIPgetProbName(self._scip)).decode('UTF-8')

    def getTotalTime(self):
        """Retrieve the current total SCIP time in seconds, i.e. the total time since the SCIP instance has been created"""
        return SCIPgetTotalTime(self._scip)

    def getSolvingTime(self):
        """Retrieve the current solving time in seconds"""
        return SCIPgetSolvingTime(self._scip)

    def getReadingTime(self):
        """Retrieve the current reading time in seconds"""
        return SCIPgetReadingTime(self._scip)

    def getPresolvingTime(self):
        """Retrieve the curernt presolving time in seconds"""
        return SCIPgetPresolvingTime(self._scip)

    def getNNodes(self):
        """Retrieve the total number of processed nodes."""
        return SCIPgetNNodes(self._scip)

    def getCurrentNode(self):
        """Retrieve current node."""
        return Node.create(SCIPgetCurrentNode(self._scip))

    def getGap(self):
        """Retrieve the gap, i.e. |(primalbound - dualbound)/min(|primalbound|,|dualbound|)|."""
        return SCIPgetGap(self._scip)

    def getDepth(self):
        """Retrieve the depth of the current node"""
        return SCIPgetDepth(self._scip)

    def infinity(self):
        """Retrieve SCIP's infinity value"""
        return SCIPinfinity(self._scip)

    def epsilon(self):
        """Retrieve epsilon for e.g. equality checks"""
        return SCIPepsilon(self._scip)

    def feastol(self):
        """Retrieve feasibility tolerance"""
        return SCIPfeastol(self._scip)

    def feasFrac(self, value):
        """returns fractional part of value, i.e. x - floor(x) in feasible tolerance: x - floor(x+feastol)"""
        return SCIPfeasFrac(self._scip, value)

    def frac(self, value):
        """returns fractional part of value, i.e. x - floor(x) in epsilon tolerance: x - floor(x+eps)"""
        return SCIPfrac(self._scip, value)

    def isZero(self, value):
        """returns whether abs(value) < eps"""
        return SCIPisZero(self._scip, value)

    def isFeasZero(self, value):
        """returns whether abs(value) < feastol"""
        return SCIPisFeasZero(self._scip, value)

    def isInfinity(self, value):
        """returns whether value is SCIP's infinity"""
        return SCIPisInfinity(self._scip, value)

    def isFeasNegative(self, value):
        """returns whether value < -feastol"""
        return SCIPisFeasNegative(self._scip, value)

    def isLE(self, val1, val2):
        """returns whether val1 <= val2 + eps"""
        return SCIPisLE(self._scip, val1, val2)

    def getCondition(self, exact=False):
        """Get the current LP's condition number

        :param exact: whether to get an estimate or the exact value (Default value = False)

        """
        cdef SCIP_LPI* lpi
        PY_SCIP_CALL(SCIPgetLPI(self._scip, &lpi))
        cdef SCIP_Real quality = 0
        if exact:
            PY_SCIP_CALL(SCIPlpiGetRealSolQuality(lpi, SCIP_LPSOLQUALITY_EXACTCONDITION, &quality))
        else:
            PY_SCIP_CALL(SCIPlpiGetRealSolQuality(lpi, SCIP_LPSOLQUALITY_ESTIMCONDITION, &quality))

        return quality

    # Objective function

    def setMinimize(self):
        """Set the objective sense to minimization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MINIMIZE))

    def setMaximize(self):
        """Set the objective sense to maximization."""
        PY_SCIP_CALL(SCIPsetObjsense(self._scip, SCIP_OBJSENSE_MAXIMIZE))

    def setObjlimit(self, objlimit):
        """Set a limit on the objective function.
        Only solutions with objective value better than this limit are accepted.

        :param objlimit: limit on the objective function

        """
        PY_SCIP_CALL(SCIPsetObjlimit(self._scip, objlimit))

    def setObjective(self, coeffs, sense = 'minimize', clear = 'true'):
        """Establish the objective function as a linear expression.

        :param coeffs: the coefficients
        :param sense: the objective sense (Default value = 'minimize')
        :param clear: set all other variables objective coefficient to zero (Default value = 'true')

        """
        cdef SCIP_VAR** _vars
        cdef int _nvars
        assert isinstance(coeffs, Expr)

        if coeffs.degree() > 1:
            raise ValueError("Nonlinear objective functions are not supported!")
        if coeffs[CONST] != 0.0:
            self.addObjoffset(coeffs[CONST])

        if clear:
            # clear existing objective function
            _vars = SCIPgetOrigVars(self._scip)
            _nvars = SCIPgetNOrigVars(self._scip)
            for i in range(_nvars):
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, _vars[i], 0.0))

        for term, coef in coeffs.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                assert len(term) == 1
                var = <Variable>term[0]
                PY_SCIP_CALL(SCIPchgVarObj(self._scip, var.var, coef))

        if sense == "minimize":
            self.setMinimize()
        elif sense == "maximize":
            self.setMaximize()
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

    def getObjective(self):
        """Retrieve objective function as Expr"""
        variables = self.getVars()
        objective = Expr()
        for var in variables:
            coeff = var.getObj()
            if coeff != 0:
                objective += coeff * var
        objective.normalize()
        return objective

    def addObjoffset(self, offset, solutions = False):
        """Add constant offset to objective

        :param offset: offset to add
        :param solutions: add offset also to existing solutions (Default value = False)

        """
        if solutions:
            PY_SCIP_CALL(SCIPaddObjoffset(self._scip, offset))
        else:
            PY_SCIP_CALL(SCIPaddOrigObjoffset(self._scip, offset))

    def getObjoffset(self, original = True):
        """Retrieve constant objective offset

        :param original: offset of original or transformed problem (Default value = True)

        """
        if original:
            return SCIPgetOrigObjoffset(self._scip)
        else:
            return SCIPgetTransObjoffset(self._scip)

    # Setting parameters
    def setPresolve(self, setting):
        """Set presolving parameter settings.

        :param setting: the parameter settings (SCIP_PARAMSETTING)

        """
        PY_SCIP_CALL(SCIPsetPresolving(self._scip, setting, True))

    def setProbName(self, name):
        """Set problem name"""
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetProbName(self._scip, n))

    def setSeparating(self, setting):
        """Set separating parameter settings.

        :param setting: the parameter settings (SCIP_PARAMSETTING)

        """
        PY_SCIP_CALL(SCIPsetSeparating(self._scip, setting, True))

    def setHeuristics(self, setting):
        """Set heuristics parameter settings.

        :param setting: the parameter setting (SCIP_PARAMSETTING)

        """
        PY_SCIP_CALL(SCIPsetHeuristics(self._scip, setting, True))

    def disablePropagation(self, onlyroot=False):
        """Disables propagation in SCIP to avoid modifying the original problem during transformation.

        :param onlyroot: use propagation when root processing is finished (Default value = False)

        """
        self.setIntParam("propagating/maxroundsroot", 0)
        if not onlyroot:
            self.setIntParam("propagating/maxrounds", 0)

    def writeProblem(self, filename='model.cip', trans=False):
        """Write current model/problem to a file.

        :param filename: the name of the file to be used (Default value = 'model.cip')
        :param trans: indicates whether the transformed problem is written to file (Default value = False)

        """
        fn = str_conversion(filename)
        fn, ext = splitext(fn)
        if len(ext) == 0:
            filename += '.cip'
            ext = str_conversion('.cip')
        ext = ext[1:]
        if trans:
            PY_SCIP_CALL(SCIPwriteTransProblem(self._scip, fn, ext, False))
        else:
            PY_SCIP_CALL(SCIPwriteOrigProblem(self._scip, fn, ext, False))
        print('wrote problem to file ' + filename)

    # Variable Functions

    def addVar(self, name='', vtype='C', lb=0.0, ub=None, obj=0.0, pricedVar = False):
        """Create a new variable. Default variable is non-negative and continuous.

        :param name: name of the variable, generic if empty (Default value = '')
        :param vtype: type of the variable (Default value = 'C')
        :param lb: lower bound of the variable, use None for -infinity (Default value = 0.0)
        :param ub: upper bound of the variable, use None for +infinity (Default value = None)
        :param obj: objective value of variable (Default value = 0.0)
        :param pricedVar: is the variable a pricing candidate? (Default value = False)

        """

        # replace empty name with generic one
        if name == '':
            name = 'x'+str(SCIPgetNVars(self._scip)+1)

        cname = str_conversion(name)
        if ub is None:
            ub = SCIPinfinity(self._scip)
        if lb is None:
            lb = -SCIPinfinity(self._scip)
        cdef SCIP_VAR* scip_var
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_CONTINUOUS))
        elif vtype in ['B', 'BINARY']:
            lb = 0.0
            ub = 1.0
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_BINARY))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPcreateVarBasic(self._scip, &scip_var, cname, lb, ub, obj, SCIP_VARTYPE_INTEGER))
        else:
            raise Warning("unrecognized variable type")

        if pricedVar:
            PY_SCIP_CALL(SCIPaddPricedVar(self._scip, scip_var, 1.0))
        else:
            PY_SCIP_CALL(SCIPaddVar(self._scip, scip_var))

        pyVar = Variable.create(scip_var)
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &scip_var))
        return pyVar

    def releaseVar(self, Variable var):
        """Release the variable.

        :param Variable var: variable to be released

        """
        PY_SCIP_CALL(SCIPreleaseVar(self._scip, &var.var))

    def getTransformedVar(self, Variable var):
        """Retrieve the transformed variable.

        :param Variable var: original variable to get the transformed of

        """
        cdef SCIP_VAR* _tvar
        PY_SCIP_CALL(SCIPtransformVar(self._scip, var.var, &_tvar))
        return Variable.create(_tvar)

    def addVarLocks(self, Variable var, nlocksdown, nlocksup):
        """adds given values to lock numbers of variable for rounding

        :param Variable var: variable to adjust the locks for
        :param nlocksdown: new number of down locks
        :param nlocksup: new number of up locks

        """
        PY_SCIP_CALL(SCIPaddVarLocks(self._scip, var.var, nlocksdown, nlocksup))

    def chgVarLb(self, Variable var, lb):
        """Changes the lower bound of the specified variable.

        :param Variable var: variable to change bound of
        :param lb: new lower bound (set to None for -infinity)

        """
        if lb is None:
           lb = -SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarLb(self._scip, var.var, lb))

    def chgVarUb(self, Variable var, ub):
        """Changes the upper bound of the specified variable.

        :param Variable var: variable to change bound of
        :param ub: new upper bound (set to None for +infinity)

        """
        if ub is None:
           ub = SCIPinfinity(self._scip)
        PY_SCIP_CALL(SCIPchgVarUb(self._scip, var.var, ub))

    def chgVarType(self, Variable var, vtype):
        """Changes the type of a variable

        :param Variable var: variable to change type of
        :param vtype: new variable type

        """
        cdef SCIP_Bool infeasible
        if vtype in ['C', 'CONTINUOUS']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.var, SCIP_VARTYPE_CONTINUOUS, &infeasible))
        elif vtype in ['B', 'BINARY']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.var, SCIP_VARTYPE_BINARY, &infeasible))
        elif vtype in ['I', 'INTEGER']:
            PY_SCIP_CALL(SCIPchgVarType(self._scip, var.var, SCIP_VARTYPE_INTEGER, &infeasible))
        else:
            raise Warning("unrecognized variable type")
        if infeasible:
            print('could not change variable type of variable %s' % var)

    def getVars(self, transformed=False):
        """Retrieve all variables.

        :param transformed: get transformed variables instead of original (Default value = False)

        """
        cdef SCIP_VAR** _vars
        cdef SCIP_VAR* _var
        cdef int _nvars
        vars = []

        if transformed:
            _vars = SCIPgetVars(self._scip)
            _nvars = SCIPgetNVars(self._scip)
        else:
            _vars = SCIPgetOrigVars(self._scip)
            _nvars = SCIPgetNOrigVars(self._scip)

        return [Variable.create(_vars[i]) for i in range(_nvars)]

    # LP Methods
    def getLPColsData(self):
        """Retrieve current LP columns"""
        cdef SCIP_COL** cols
        cdef int ncols

        PY_SCIP_CALL(SCIPgetLPColsData(self._scip, &cols, &ncols))
        return [Column.create(cols[i]) for i in range(ncols)]

    def getLPRowsData(self):
        """Retrieve current LP rows"""
        cdef SCIP_ROW** rows
        cdef int nrows

        PY_SCIP_CALL(SCIPgetLPRowsData(self._scip, &rows, &nrows))
        return [Row.create(rows[i]) for i in range(nrows)]

    def getLPBasisInd(self):
        """Gets all indices of basic columns and rows: index i >= 0 corresponds to column i, index i < 0 to row -i-1"""
        cdef int nrows = SCIPgetNLPRows(self._scip)
        cdef int* inds = <int *> malloc(nrows * sizeof(int))

        PY_SCIP_CALL(SCIPgetLPBasisInd(self._scip, inds))
        result = [inds[i] for i in range(nrows)]
        free(inds)
        return result

    def getLPBInvRow(self, row):
        """gets a row from the inverse basis matrix B^-1"""
        # TODO: sparsity information
        cdef int nrows = SCIPgetNLPRows(self._scip)
        cdef SCIP_Real* coefs = <SCIP_Real*> malloc(nrows * sizeof(SCIP_Real))

        PY_SCIP_CALL(SCIPgetLPBInvRow(self._scip, row, coefs, NULL, NULL))
        result = [coefs[i] for i in range(nrows)]
        free(coefs)
        return result

    def getLPBInvARow(self, row):
        """gets a row from B^-1 * A"""
        # TODO: sparsity information
        cdef int ncols = SCIPgetNLPCols(self._scip)
        cdef SCIP_Real* coefs = <SCIP_Real*> malloc(ncols * sizeof(SCIP_Real))

        PY_SCIP_CALL(SCIPgetLPBInvARow(self._scip, row, NULL, coefs, NULL, NULL))
        result = [coefs[i] for i in range(ncols)]
        free(coefs)
        return result

    def isLPSolBasic(self):
        """returns whether the current LP solution is basic, i.e. is defined by a valid simplex basis"""
        return SCIPisLPSolBasic(self._scip)

    #TODO: documentation!!
    # LP Row Methods
    def createEmptyRowSepa(self, Sepa sepa, name="row", lhs = 0.0, rhs = None, local = True, modifiable = False, removable = True):
        cdef SCIP_ROW* row
        lhs =  -SCIPinfinity(self._scip) if lhs is None else lhs
        rhs =  SCIPinfinity(self._scip) if rhs is None else rhs
        scip_sepa = SCIPfindSepa(self._scip, str_conversion(sepa.name))
        PY_SCIP_CALL(SCIPcreateEmptyRowSepa(self._scip, &row, scip_sepa, str_conversion(name), lhs, rhs, local, modifiable, removable))
        PyRow = Row.create(row)
        return PyRow

    def getRowActivity(self, Row row):
        return SCIPgetRowActivity(self._scip, row.row)

    def getRowLPActivity(self, Row row):
        return SCIPgetRowLPActivity(self._scip, row.row)

    # TODO: do we need this? (also do we need release var??)
    def releaseRow(self, Row row not None):
        PY_SCIP_CALL(SCIPreleaseRow(self._scip, &row.row))

    def cacheRowExtensions(self, Row row not None):
        PY_SCIP_CALL(SCIPcacheRowExtensions(self._scip, row.row))

    def flushRowExtensions(self, Row row not None):
        PY_SCIP_CALL(SCIPflushRowExtensions(self._scip, row.row))

    def addVarToRow(self, Row row not None, Variable var not None, value):
        PY_SCIP_CALL(SCIPaddVarToRow(self._scip, row.row, var.var, value))

    def printRow(self, Row row not None):
        """Prints row."""
        PY_SCIP_CALL(SCIPprintRow(self._scip, row.row, NULL))

    # Cutting Plane Methods
    def addPoolCut(self, Row row not None):
        PY_SCIP_CALL(SCIPaddPoolCut(self._scip, row.row))

    def getCutEfficacy(self, Row cut not None, Solution sol = None):
        return SCIPgetCutEfficacy(self._scip, NULL if sol is None else sol.sol, cut.row)

    def isCutEfficacious(self, Row cut not None, Solution sol = None):
        """ returns whether the cut's efficacy with respect to the given primal solution or the current LP solution is greater than the minimal cut efficacy"""
        return SCIPisCutEfficacious(self._scip, NULL if sol is None else sol.sol, cut.row)

    def addCut(self, Row cut not None, forcecut = False):
        """adds cut to separation storage and returns whether cut has been detected to be infeasible for local bounds"""
        cdef SCIP_Bool infeasible
        PY_SCIP_CALL(SCIPaddRow(self._scip, cut.row, forcecut, &infeasible))
        return infeasible

    # Constraint functions
    def addCons(self, cons, name='', initial=True, separate=True,
                enforce=True, check=True, propagate=True, local=False,
                modifiable=False, dynamic=False, removable=False,
                stickingatnode=False):
        """Add a linear or quadratic constraint.

        :param cons: list of coefficients
        :param name: the name of the constraint, generic name if empty (Default value = '')
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked during for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param modifiable: is the constraint modifiable (subject to column generation)? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be  moved to a more global node? (Default value = False)

        """
        assert isinstance(cons, ExprCons)

        # replace empty name with generic one
        if name == '':
            name = 'c'+str(SCIPgetNConss(self._scip)+1)

        kwargs = dict(name=name, initial=initial, separate=separate,
                      enforce=enforce, check=check,
                      propagate=propagate, local=local,
                      modifiable=modifiable, dynamic=dynamic,
                      removable=removable,
                      stickingatnode=stickingatnode)
        kwargs['lhs'] = -SCIPinfinity(self._scip) if cons.lhs is None else cons.lhs
        kwargs['rhs'] =  SCIPinfinity(self._scip) if cons.rhs is None else cons.rhs

        deg = cons.expr.degree()
        if deg <= 1:
            return self._addLinCons(cons, **kwargs)
        elif deg <= 2:
            return self._addQuadCons(cons, **kwargs)
        elif deg == float('inf'): # general nonlinear
            return self._addGenNonlinearCons(cons, **kwargs)
        else:
            return self._addNonlinearCons(cons, **kwargs)

    def _addLinCons(self, ExprCons lincons, **kwargs):
        assert isinstance(lincons, ExprCons)

        assert lincons.expr.degree() <= 1
        terms = lincons.expr.terms

        cdef SCIP_CONS* scip_cons
        PY_SCIP_CALL(SCIPcreateConsLinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']), 0, NULL, NULL,
            kwargs['lhs'], kwargs['rhs'], kwargs['initial'],
            kwargs['separate'], kwargs['enforce'], kwargs['check'],
            kwargs['propagate'], kwargs['local'], kwargs['modifiable'],
            kwargs['dynamic'], kwargs['removable'], kwargs['stickingatnode']))

        for key, coeff in terms.items():
            var = <Variable>key[0]
            PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, scip_cons, var.var, <SCIP_Real>coeff))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return PyCons

    def _addQuadCons(self, ExprCons quadcons, **kwargs):
        terms = quadcons.expr.terms
        assert quadcons.expr.degree() <= 2

        cdef SCIP_CONS* scip_cons
        PY_SCIP_CALL(SCIPcreateConsQuadratic(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL,        # linear
            0, NULL, NULL, NULL,  # quadratc
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable']))

        for v, c in terms.items():
            if len(v) == 1: # linear
                var = <Variable>v[0]
                PY_SCIP_CALL(SCIPaddLinearVarQuadratic(self._scip, scip_cons, var.var, c))
            else: # quadratic
                assert len(v) == 2, 'term: %s' % v
                var1, var2 = <Variable>v[0], <Variable>v[1]
                PY_SCIP_CALL(SCIPaddBilinTermQuadratic(self._scip, scip_cons, var1.var, var2.var, c))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        return PyCons

    def _addNonlinearCons(self, ExprCons cons, **kwargs):
        cdef SCIP_EXPR* expr
        cdef SCIP_EXPR** varexprs
        cdef SCIP_EXPRDATA_MONOMIAL** monomials
        cdef int* idxs
        cdef SCIP_EXPRTREE* exprtree
        cdef SCIP_VAR** vars
        cdef SCIP_CONS* scip_cons

        terms = cons.expr.terms

        # collect variables
        variables = {var.ptr():var for term in terms for var in term}
        variables = list(variables.values())
        varindex = {var.ptr():idx for (idx,var) in enumerate(variables)}

        # create variable expressions
        varexprs = <SCIP_EXPR**> malloc(len(varindex) * sizeof(SCIP_EXPR*))
        for idx in varindex.values():
            PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &expr, SCIP_EXPR_VARIDX, <int>idx) )
            varexprs[idx] = expr

        # create monomials for terms
        monomials = <SCIP_EXPRDATA_MONOMIAL**> malloc(len(terms) * sizeof(SCIP_EXPRDATA_MONOMIAL*))
        for i, (term, coef) in enumerate(terms.items()):
            idxs = <int*> malloc(len(term) * sizeof(int))
            for j, var in enumerate(term):
                idxs[j] = varindex[var.ptr()]
            PY_SCIP_CALL( SCIPexprCreateMonomial(SCIPblkmem(self._scip), &monomials[i], <SCIP_Real>coef, <int>len(term), idxs, NULL) )
            free(idxs)

        # create polynomial from monomials
        PY_SCIP_CALL( SCIPexprCreatePolynomial(SCIPblkmem(self._scip), &expr,
                                               <int>len(varindex), varexprs,
                                               <int>len(terms), monomials, 0.0, <SCIP_Bool>True) )

        # create expression tree
        PY_SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(self._scip), &exprtree, expr, <int>len(variables), 0, NULL) )
        vars = <SCIP_VAR**> malloc(len(variables) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(variables): # same as varindex
            vars[idx] = (<Variable>var).var
        PY_SCIP_CALL( SCIPexprtreeSetVars(exprtree, <int>len(variables), vars) )

        # create nonlinear constraint for exprtree
        PY_SCIP_CALL( SCIPcreateConsNonlinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL, # linear
            1, &exprtree, NULL, # nonlinear
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable'],
            kwargs['stickingatnode']) )
        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        PY_SCIP_CALL( SCIPexprtreeFree(&exprtree) )
        free(vars)
        free(monomials)
        free(varexprs)
        return PyCons

    def _addGenNonlinearCons(self, ExprCons cons, **kwargs):
        cdef SCIP_EXPR** childrenexpr
        cdef SCIP_EXPR** scipexprs
        cdef SCIP_EXPRTREE* exprtree
        cdef SCIP_CONS* scip_cons
        cdef int nchildren

        # get arrays from python's expression tree
        expr = cons.expr
        nodes = expr_to_nodes(expr)
        op2idx = Operator.operatorIndexDic

        # in nodes we have a list of tuples: each tuple is of the form
        # (operator, [indices]) where indices are the indices of the tuples
        # that are the children of this operator. This is sorted,
        # so we are going to do is:
        # loop over the nodes and create the expression of each
        # Note1: when the operator is SCIP_EXPR_CONST, [indices] stores the value
        # Note2: we need to compute the number of variable operators to find out
        # how many variables are there.
        nvars = 0
        for node in nodes:
            if op2idx[node[0]] == SCIP_EXPR_VARIDX:
                nvars += 1
        vars = <SCIP_VAR**> malloc(nvars * sizeof(SCIP_VAR*))

        varpos = 0
        scipexprs = <SCIP_EXPR**> malloc(len(nodes) * sizeof(SCIP_EXPR*))
        for i,node in enumerate(nodes):
            op = node[0]
            opidx = op2idx[op]
            if opidx == SCIP_EXPR_VARIDX:
                assert len(node[1]) == 1
                pyvar = node[1][0] # for vars we store the actual var!
                PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, <int>varpos) )
                vars[varpos] = (<Variable>pyvar).var
                varpos += 1
                continue
            if opidx == SCIP_EXPR_CONST:
                assert len(node[1]) == 1
                value = node[1][0]
                PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, <SCIP_Real>value) )
                continue
            if opidx == SCIP_EXPR_SUM or opidx == SCIP_EXPR_PRODUCT:
                nchildren = len(node[1])
                childrenexpr = <SCIP_EXPR**> malloc(nchildren * sizeof(SCIP_EXPR*))
                for c, pos in enumerate(node[1]):
                    childrenexpr[c] = scipexprs[pos]
                PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, nchildren, childrenexpr) )

                free(childrenexpr);
                continue
            if opidx == SCIP_EXPR_REALPOWER:
                # the second child is the exponent which is a const
                valuenode = nodes[node[1][1]]
                assert op2idx[valuenode[0]] == SCIP_EXPR_CONST
                exponent = valuenode[1][0]
                if float(exponent).is_integer():
                    PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], SCIP_EXPR_INTPOWER, scipexprs[node[1][0]], <int>exponent) )
                else:
                    PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, scipexprs[node[1][0]], <SCIP_Real>exponent) )
                continue
            if opidx == SCIP_EXPR_EXP or opidx == SCIP_EXPR_LOG or opidx == SCIP_EXPR_SQRT or opidx == SCIP_EXPR_ABS:
                assert len(node[1]) == 1
                PY_SCIP_CALL( SCIPexprCreate(SCIPblkmem(self._scip), &scipexprs[i], opidx, scipexprs[node[1][0]]) )
                continue
            # default:
            raise NotImplementedError
        assert varpos == nvars

        # create expression tree
        PY_SCIP_CALL( SCIPexprtreeCreate(SCIPblkmem(self._scip), &exprtree, scipexprs[len(nodes) - 1], nvars, 0, NULL) );
        PY_SCIP_CALL( SCIPexprtreeSetVars(exprtree, <int>nvars, vars) );

        # create nonlinear constraint for exprtree
        PY_SCIP_CALL( SCIPcreateConsNonlinear(
            self._scip, &scip_cons, str_conversion(kwargs['name']),
            0, NULL, NULL, # linear
            1, &exprtree, NULL, # nonlinear
            kwargs['lhs'], kwargs['rhs'],
            kwargs['initial'], kwargs['separate'], kwargs['enforce'],
            kwargs['check'], kwargs['propagate'], kwargs['local'],
            kwargs['modifiable'], kwargs['dynamic'], kwargs['removable'],
            kwargs['stickingatnode']) )
        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        PyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))
        PY_SCIP_CALL( SCIPexprtreeFree(&exprtree) )

        # free more memory
        free(scipexprs)
        free(vars)

        return PyCons

    def addConsCoeff(self, Constraint cons, Variable var, coeff):
        """Add coefficient to the linear constraint (if non-zero).

        :param Constraint cons: constraint to be changed
        :param Variable var: variable to be added
        :param coeff: coefficient of new variable

        """
        PY_SCIP_CALL(SCIPaddCoefLinear(self._scip, cons.cons, var.var, coeff))

    def addConsSOS1(self, vars, weights=None, name="SOS1cons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an SOS1 constraint.

        :param vars: list of variables to be included
        :param weights: list of weights (Default value = None)
        :param name: name of the constraint (Default value = "SOS1cons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)

        """
        cdef SCIP_CONS* scip_cons
        cdef int _nvars

        PY_SCIP_CALL(SCIPcreateConsSOS1(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if weights is None:
            for v in vars:
                var = <Variable>v
                PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, scip_cons, var.var))
        else:
            nvars = len(vars)
            for i in range(nvars):
                var = <Variable>vars[i]
                PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, scip_cons, var.var, weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        return Constraint.create(scip_cons)

    def addConsSOS2(self, vars, weights=None, name="SOS2cons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an SOS2 constraint.

        :param vars: list of variables to be included
        :param weights: list of weights (Default value = None)
        :param name: name of the constraint (Default value = "SOS2cons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: is the constraint only valid locally? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)

        """
        cdef SCIP_CONS* scip_cons
        cdef int _nvars

        PY_SCIP_CALL(SCIPcreateConsSOS2(self._scip, &scip_cons, str_conversion(name), 0, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        if weights is None:
            for v in vars:
                var = <Variable>v
                PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, scip_cons, var.var))
        else:
            nvars = len(vars)
            for i in range(nvars):
                var = <Variable>vars[i]
                PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, scip_cons, var.var, weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        return Constraint.create(scip_cons)

    def addConsAnd(self, vars, resvar, name="ANDcons",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """Add an AND-constraint.
        :param vars: list of BINARY variables to be included (operators)
        :param resvar: BINARY variable (resultant)
        :param name: name of the constraint (Default value = "ANDcons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param modifiable: is the constraint modifiable (subject to column generation)? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)
        """
        cdef SCIP_CONS* scip_cons

        nvars = len(vars)

        _vars = <SCIP_VAR**> malloc(len(vars) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(vars):
            _vars[idx] = (<Variable>var).var
        _resVar = (<Variable>resvar).var

        PY_SCIP_CALL(SCIPcreateConsAnd(self._scip, &scip_cons, str_conversion(name), _resVar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsOr(self, vars, resvar, name="ORcons",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """Add an OR-constraint.
        :param vars: list of BINARY variables to be included (operators)
        :param resvar: BINARY variable (resultant)
        :param name: name of the constraint (Default value = "ORcons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param modifiable: is the constraint modifiable (subject to column generation)? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)
        """
        cdef SCIP_CONS* scip_cons

        nvars = len(vars)

        _vars = <SCIP_VAR**> malloc(len(vars) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(vars):
            _vars[idx] = (<Variable>var).var
        _resVar = (<Variable>resvar).var

        PY_SCIP_CALL(SCIPcreateConsOr(self._scip, &scip_cons, str_conversion(name), _resVar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsXor(self, vars, rhsvar, name="XORcons",
            initial=True, separate=True, enforce=True, check=True,
            propagate=True, local=False, modifiable=False, dynamic=False,
            removable=False, stickingatnode=False):
        """Add a XOR-constraint.
        :param vars: list of BINARY variables to be included (operators)
        :param rhsvar: BOOLEAN value, explicit True, False or bool(obj) is needed (right-hand side)
        :param name: name of the constraint (Default value = "XORcons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param modifiable: is the constraint modifiable (subject to column generation)? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)
        """
        cdef SCIP_CONS* scip_cons

        nvars = len(vars)

        assert type(rhsvar) is type(bool()), "Provide BOOLEAN value as rhsvar, you gave %s." % type(rhsvar)
        _vars = <SCIP_VAR**> malloc(len(vars) * sizeof(SCIP_VAR*))
        for idx, var in enumerate(vars):
            _vars[idx] = (<Variable>var).var

        PY_SCIP_CALL(SCIPcreateConsXor(self._scip, &scip_cons, str_conversion(name), rhsvar, nvars, _vars,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)
        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        free(_vars)

        return pyCons

    def addConsCardinality(self, consvars, cardval, indvars=None, weights=None, name="CardinalityCons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add a cardinality constraint that allows at most 'cardval' many nonzero variables.

        :param consvars: list of variables to be included
        :param cardval: nonnegative integer
        :param indvars: indicator variables indicating which variables may be treated as nonzero in cardinality constraint, or None if new indicator variables should be introduced automatically (Default value = None)
        :param weights: weights determining the variable order, or None if variables should be ordered in the same way they were added to the constraint (Default value = None)
        :param name: name of the constraint (Default value = "CardinalityCons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)

        """
        cdef SCIP_CONS* scip_cons
        cdef SCIP_VAR* indvar

        PY_SCIP_CALL(SCIPcreateConsCardinality(self._scip, &scip_cons, str_conversion(name), 0, NULL, cardval, NULL, NULL,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))

        # circumvent an annoying bug in SCIP 4.0.0 that does not allow uninitialized weights
        if weights is None:
            weights = list(range(1, len(consvars) + 1))

        for i, v in enumerate(consvars):
            var = <Variable>v
            if indvars:
                indvar = (<Variable>indvars[i]).var
            else:
                indvar = NULL
            if weights is None:
                PY_SCIP_CALL(SCIPappendVarCardinality(self._scip, scip_cons, var.var, indvar))
            else:
                PY_SCIP_CALL(SCIPaddVarCardinality(self._scip, scip_cons, var.var, indvar, <SCIP_Real>weights[i]))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons


    def addConsIndicator(self, cons, binvar=None, name="CardinalityCons",
                initial=True, separate=True, enforce=True, check=True,
                propagate=True, local=False, dynamic=False,
                removable=False, stickingatnode=False):
        """Add an indicator constraint for the linear inequality 'cons'.

        The 'binvar' argument models the redundancy of the linear constraint. A solution for which
        'binvar' is 1 must satisfy the constraint.

        :param cons: a linear inequality of the form "<="
        :param binvar: binary indicator variable, or None if it should be created (Default value = None)
        :param name: name of the constraint (Default value = "CardinalityCons")
        :param initial: should the LP relaxation of constraint be in the initial LP? (Default value = True)
        :param separate: should the constraint be separated during LP processing? (Default value = True)
        :param enforce: should the constraint be enforced during node processing? (Default value = True)
        :param check: should the constraint be checked for feasibility? (Default value = True)
        :param propagate: should the constraint be propagated during node processing? (Default value = True)
        :param local: is the constraint only valid locally? (Default value = False)
        :param dynamic: is the constraint subject to aging? (Default value = False)
        :param removable: should the relaxation be removed from the LP due to aging or cleanup? (Default value = False)
        :param stickingatnode: should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? (Default value = False)

        """
        assert isinstance(cons, ExprCons)
        cdef SCIP_CONS* scip_cons
        cdef SCIP_VAR* _binVar
        if cons.lhs is not None and cons.rhs is not None:
            raise ValueError("expected inequality that has either only a left or right hand side")

        if cons.expr.degree() > 1:
            raise ValueError("expected linear inequality, expression has degree %d" % cons.expr.degree())

        assert cons.expr.degree() <= 1

        if cons.rhs is not None:
            rhs =  cons.rhs
            negate = False
        else:
            rhs = -cons.lhs
            negate = True

        _binVar = (<Variable>binvar).var if binvar is not None else NULL

        PY_SCIP_CALL(SCIPcreateConsIndicator(self._scip, &scip_cons, str_conversion(name), _binVar, 0, NULL, NULL, rhs,
            initial, separate, enforce, check, propagate, local, dynamic, removable, stickingatnode))
        terms = cons.expr.terms

        for key, coeff in terms.items():
            var = <Variable>key[0]
            if negate:
                coeff = -coeff
            PY_SCIP_CALL(SCIPaddVarIndicator(self._scip, scip_cons, var.var, <SCIP_Real>coeff))

        PY_SCIP_CALL(SCIPaddCons(self._scip, scip_cons))
        pyCons = Constraint.create(scip_cons)

        PY_SCIP_CALL(SCIPreleaseCons(self._scip, &scip_cons))

        return pyCons

    def addPyCons(self, Constraint cons):
        """Adds a customly created cons.

        :param Constraint cons: constraint to add

        """
        PY_SCIP_CALL(SCIPaddCons(self._scip, cons.cons))
        Py_INCREF(cons)

    def addVarSOS1(self, Constraint cons, Variable var, weight):
        """Add variable to SOS1 constraint.

        :param Constraint cons: SOS1 constraint
        :param Variable var: new variable
        :param weight: weight of new variable

        """
        PY_SCIP_CALL(SCIPaddVarSOS1(self._scip, cons.cons, var.var, weight))

    def appendVarSOS1(self, Constraint cons, Variable var):
        """Append variable to SOS1 constraint.

        :param Constraint cons: SOS1 constraint
        :param Variable var: variable to append

        """
        PY_SCIP_CALL(SCIPappendVarSOS1(self._scip, cons.cons, var.var))

    def addVarSOS2(self, Constraint cons, Variable var, weight):
        """Add variable to SOS2 constraint.

        :param Constraint cons: SOS2 constraint
        :param Variable var: new variable
        :param weight: weight of new variable

        """
        PY_SCIP_CALL(SCIPaddVarSOS2(self._scip, cons.cons, var.var, weight))

    def appendVarSOS2(self, Constraint cons, Variable var):
        """Append variable to SOS2 constraint.

        :param Constraint cons: SOS2 constraint
        :param Variable var: variable to append

        """
        PY_SCIP_CALL(SCIPappendVarSOS2(self._scip, cons.cons, var.var))

    def setInitial(self, Constraint cons, newInit):
        """Set "initial" flag of a constraint.

        Keyword arguments:
        cons -- constraint
        newInit -- new initial value
        """
        PY_SCIP_CALL(SCIPsetConsInitial(self._scip, cons.cons, newInit))

    def setRemovable(self, Constraint cons, newRem):
        """Set "removable" flag of a constraint.

        Keyword arguments:
        cons -- constraint
        newRem -- new removable value
        """
        PY_SCIP_CALL(SCIPsetConsRemovable(self._scip, cons.cons, newRem))

    def setEnforced(self, Constraint cons, newEnf):
        """Set "enforced" flag of a constraint.

        Keyword arguments:
        cons -- constraint
        newEnf -- new enforced value
        """
        PY_SCIP_CALL(SCIPsetConsEnforced(self._scip, cons.cons, newEnf))

    def setCheck(self, Constraint cons, newCheck):
        """Set "check" flag of a constraint.

        Keyword arguments:
        cons -- constraint
        newCheck -- new check value
        """
        PY_SCIP_CALL(SCIPsetConsChecked(self._scip, cons.cons, newCheck))

    def chgRhs(self, Constraint cons, rhs):
        """Change right hand side value of a constraint.

        :param Constraint cons: linear or quadratic constraint
        :param rhs: new ride hand side (set to None for +infinity)

        """

        if rhs is None:
           rhs = SCIPinfinity(self._scip)

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgRhsLinear(self._scip, cons.cons, rhs))
        elif constype == 'quadratic':
            PY_SCIP_CALL(SCIPchgRhsQuadratic(self._scip, cons.cons, rhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def chgLhs(self, Constraint cons, lhs):
        """Change left hand side value of a constraint.

        :param Constraint cons: linear or quadratic constraint
        :param lhs: new left hand side (set to None for -infinity)

        """

        if lhs is None:
           lhs = -SCIPinfinity(self._scip)

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if constype == 'linear':
            PY_SCIP_CALL(SCIPchgLhsLinear(self._scip, cons.cons, lhs))
        elif constype == 'quadratic':
            PY_SCIP_CALL(SCIPchgLhsQuadratic(self._scip, cons.cons, lhs))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getRhs(self, Constraint cons):
        """Retrieve right hand side value of a constraint.

        :param Constraint cons: linear or quadratic constraint

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if constype == 'linear':
            return SCIPgetRhsLinear(self._scip, cons.cons)
        elif constype == 'quadratic':
            return SCIPgetRhsQuadratic(self._scip, cons.cons)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getLhs(self, Constraint cons):
        """Retrieve left hand side value of a constraint.

        :param Constraint cons: linear or quadratic constraint

        """
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if constype == 'linear':
            return SCIPgetLhsLinear(self._scip, cons.cons)
        elif constype == 'quadratic':
            return SCIPgetLhsQuadratic(self._scip, cons.cons)
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

    def getActivity(self, Constraint cons, Solution sol = None):
        """Retrieve activity of given constraint.
        Can only be called after solving is completed.

        :param Constraint cons: linear or quadratic constraint
        :param Solution sol: solution to compute activity of, None to use current node's solution (Default value = None)

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* scip_sol

        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")

        if isinstance(sol, Solution):
            scip_sol = sol.sol
        else:
            scip_sol = NULL

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if constype == 'linear':
            activity = SCIPgetActivityLinear(self._scip, cons.cons, scip_sol)
        elif constype == 'quadratic':
            PY_SCIP_CALL(SCIPgetActivityQuadratic(self._scip, cons.cons, scip_sol, &activity))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

        return activity


    def getSlack(self, Constraint cons, Solution sol = None, side = None):
        """Retrieve slack of given constraint.
        Can only be called after solving is completed.


        :param Constraint cons: linear or quadratic constraint
        :param Solution sol: solution to compute slack of, None to use current node's solution (Default value = None)
        :param side: whether to use 'lhs' or 'rhs' for ranged constraints, None to return minimum (Default value = None)

        """
        cdef SCIP_Real activity
        cdef SCIP_SOL* scip_sol


        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")

        if isinstance(sol, Solution):
            scip_sol = sol.sol
        else:
            scip_sol = NULL

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if constype == 'linear':
            lhs = SCIPgetLhsLinear(self._scip, cons.cons)
            rhs = SCIPgetRhsLinear(self._scip, cons.cons)
            activity = SCIPgetActivityLinear(self._scip, cons.cons, scip_sol)
        elif constype == 'quadratic':
            lhs = SCIPgetLhsQuadratic(self._scip, cons.cons)
            rhs = SCIPgetRhsQuadratic(self._scip, cons.cons)
            PY_SCIP_CALL(SCIPgetActivityQuadratic(self._scip, cons.cons, scip_sol, &activity))
        else:
            raise Warning("method cannot be called for constraints of type " + constype)

        lhsslack = activity - lhs
        rhsslack = rhs - activity

        if side == 'lhs':
            return lhsslack
        elif side == 'rhs':
            return rhsslack
        else:
            return min(lhsslack, rhsslack)

    def getTransformedCons(self, Constraint cons):
        """Retrieve transformed constraint.

        :param Constraint cons: constraint

        """
        cdef SCIP_CONS* transcons
        PY_SCIP_CALL(SCIPgetTransformedCons(self._scip, cons.cons, &transcons))
        return Constraint.create(transcons)

    def getTermsQuadratic(self, Constraint cons):
        """Retrieve bilinear, quadratic, and linear terms of a quadratic constraint.

        :param Constraint cons: constraint

        """
        cdef SCIP_QUADVARTERM* _quadterms
        cdef SCIP_BILINTERM* _bilinterms
        cdef SCIP_VAR** _linvars
        cdef SCIP_Real* _lincoefs
        cdef int _nbilinterms
        cdef int _nquadterms
        cdef int _nlinvars

        assert cons.isQuadratic()

        bilinterms = []
        quadterms  = []
        linterms   = []

        # bilinear terms
        _bilinterms = SCIPgetBilinTermsQuadratic(self._scip, cons.cons)
        _nbilinterms = SCIPgetNBilinTermsQuadratic(self._scip, cons.cons)

        for i in range(_nbilinterms):
            var1 = Variable.create(_bilinterms[i].var1)
            var2 = Variable.create(_bilinterms[i].var2)
            bilinterms.append((var1,var2,_bilinterms[i].coef))

        # quadratic terms
        _quadterms = SCIPgetQuadVarTermsQuadratic(self._scip, cons.cons)
        _nquadterms = SCIPgetNQuadVarTermsQuadratic(self._scip, cons.cons)

        for i in range(_nquadterms):
            var = Variable.create(_quadterms[i].var)
            quadterms.append((var,_quadterms[i].sqrcoef,_quadterms[i].lincoef))

        # linear terms
        _linvars = SCIPgetLinearVarsQuadratic(self._scip, cons.cons)
        _lincoefs = SCIPgetCoefsLinearVarsQuadratic(self._scip, cons.cons)
        _nlinvars = SCIPgetNLinearVarsQuadratic(self._scip, cons.cons)

        for i in range(_nlinvars):
            var = Variable.create(_linvars[i])
            linterms.append((var,_lincoefs[i]))

        return (bilinterms, quadterms, linterms)

    def getConss(self):
        """Retrieve all constraints."""
        cdef SCIP_CONS** _conss
        cdef SCIP_CONS* _cons
        cdef int _nconss
        conss = []

        _conss = SCIPgetConss(self._scip)
        _nconss = SCIPgetNConss(self._scip)
        return [Constraint.create(_conss[i]) for i in range(_nconss)]

    def delCons(self, Constraint cons):
        """Delete constraint from the model

        :param Constraint cons: constraint to be deleted

        """
        PY_SCIP_CALL(SCIPdelCons(self._scip, cons.cons))

    def delConsLocal(self, Constraint cons):
        """Delete constraint from the current node and it's children

        :param Constraint cons: constraint to be deleted

        """
        PY_SCIP_CALL(SCIPdelConsLocal(self._scip, cons.cons))

    def getValsLinear(self, Constraint cons):
        """Retrieve the coefficients of a linear constraint

        :param Constraint cons: linear constraint to get the coefficients of

        """
        cdef SCIP_Real* _vals
        cdef SCIP_VAR** _vars

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("coefficients not available for constraints of type ", constype)

        _vals = SCIPgetValsLinear(self._scip, cons.cons)
        _vars = SCIPgetVarsLinear(self._scip, cons.cons)

        valsdict = {}
        for i in range(SCIPgetNVarsLinear(self._scip, cons.cons)):
            valsdict[bytes(SCIPvarGetName(_vars[i])).decode('utf-8')] = _vals[i]
        return valsdict

    def getDualsolLinear(self, Constraint cons):
        """Retrieve the dual solution to a linear constraint.

        :param Constraint cons: linear constraint

        """
        # TODO this should ideally be handled on the SCIP side
        cdef int _nvars
        cdef SCIP_VAR** _vars
        cdef SCIP_Bool _success
        dual = 0.0

        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(cons.cons))).decode('UTF-8')
        if not constype == 'linear':
            raise Warning("dual solution values not available for constraints of type ", constype)

        try:
            _nvars = SCIPgetNVarsLinear(self._scip, cons.cons)
            if cons.isOriginal():
                transcons = <Constraint>self.getTransformedCons(cons)
            else:
                transcons = cons
            dual = SCIPgetDualsolLinear(self._scip, transcons.cons)
            if dual == 0.0 and _nvars == 1:
                _vars = SCIPgetVarsLinear(self._scip, transcons.cons)
                LPsol = SCIPvarGetLPSol(_vars[0])
                rhs = SCIPgetRhsLinear(self._scip, transcons.cons)
                lhs = SCIPgetLhsLinear(self._scip, transcons.cons)
                if (LPsol == rhs) or (LPsol == lhs):
                    dual = SCIPgetVarRedcost(self._scip, _vars[0])

            if self.getObjectiveSense() == "maximize":
                dual = -dual
        except:
            raise Warning("no dual solution available for constraint " + cons.name)
        return dual

    def getDualfarkasLinear(self, Constraint cons):
        """Retrieve the dual farkas value to a linear constraint.

        :param Constraint cons: linear constraint

        """
        # TODO this should ideally be handled on the SCIP side
        if cons.isOriginal():
            transcons = <Constraint>self.getTransformedCons(cons)
            return SCIPgetDualfarkasLinear(self._scip, transcons.cons)
        else:
            return SCIPgetDualfarkasLinear(self._scip, cons.cons)

    def getVarRedcost(self, Variable var):
        """Retrieve the reduced cost of a variable.

        :param Variable var: variable to get the reduced cost of

        """
        redcost = None
        try:
            redcost = SCIPgetVarRedcost(self._scip, var.var)
            if self.getObjectiveSense() == "maximize":
                redcost = -redcost
        except:
            raise Warning("no reduced cost available for variable " + var.name)
        return redcost

    def optimize(self):
        """Optimize the problem."""
        PY_SCIP_CALL(SCIPsolve(self._scip))
        self._bestSol = Solution.create(SCIPgetBestSol(self._scip))

    def presolve(self):
        """Presolve the problem."""
        PY_SCIP_CALL(SCIPpresolve(self._scip))

    def interrupt(self):
        """Interrupts the solving process."""
        PY_SCIP_CALL(SCIPinterruptSolve(self._scip))

    def includeEventhdlr(self, Eventhdlr eventhdlr, name, desc):
        """Include an event handler.

        Keyword arguments:
        eventhdlr -- event handler
        name -- name of event handler
        desc -- description of event handler
        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeEventhdlr(self._scip, n, d,
                                          PyEventCopy,
                                          PyEventFree,
                                          PyEventInit,
                                          PyEventExit,
                                          PyEventInitsol,
                                          PyEventExitsol,
                                          PyEventDelete,
                                          PyEventExec,
                                          <SCIP_EVENTHDLRDATA*>eventhdlr))
        eventhdlr.model = <Model>weakref.proxy(self)
        eventhdlr.name = name
        Py_INCREF(eventhdlr)

    def includePricer(self, Pricer pricer, name, desc, priority=1, delay=True):
        """Include a pricer.

        :param Pricer pricer: pricer
        :param name: name of pricer
        :param desc: description of pricer
        :param priority: priority of pricer (Default value = 1)
        :param delay: should the pricer be delayed until no other pricers or already existing problem variables with negative reduced costs are found? (Default value = True)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludePricer(self._scip, n, d,
                                            priority, delay,
                                            PyPricerCopy, PyPricerFree, PyPricerInit, PyPricerExit, PyPricerInitsol, PyPricerExitsol, PyPricerRedcost, PyPricerFarkas,
                                            <SCIP_PRICERDATA*>pricer))
        cdef SCIP_PRICER* scip_pricer
        scip_pricer = SCIPfindPricer(self._scip, n)
        PY_SCIP_CALL(SCIPactivatePricer(self._scip, scip_pricer))
        pricer.model = <Model>weakref.proxy(self)
        Py_INCREF(pricer)

    def includeConshdlr(self, Conshdlr conshdlr, name, desc, sepapriority=0,
                        enfopriority=0, chckpriority=0, sepafreq=-1, propfreq=-1,
                        eagerfreq=100, maxprerounds=-1, delaysepa=False,
                        delayprop=False, needscons=True,
                        proptiming=PY_SCIP_PROPTIMING.BEFORELP,
                        presoltiming=PY_SCIP_PRESOLTIMING.MEDIUM):
        """Include a constraint handler

        :param Conshdlr conshdlr: constraint handler
        :param name: name of constraint handler
        :param desc: description of constraint handler
        :param sepapriority: priority for separation (Default value = 0)
        :param enfopriority: priority for constraint enforcing (Default value = 0)
        :param chckpriority: priority for checking feasibility (Default value = 0)
        :param sepafreq: frequency for separating cuts; 0 = only at root node (Default value = -1)
        :param propfreq: frequency for propagating domains; 0 = only preprocessing propagation (Default value = -1)
        :param eagerfreq: frequency for using all instead of only the useful constraints in separation, propagation and enforcement; -1 = no eager evaluations, 0 = first only (Default value = 100)
        :param maxprerounds: maximal number of presolving rounds the constraint handler participates in (Default value = -1)
        :param delaysepa: should separation method be delayed, if other separators found cuts? (Default value = False)
        :param delayprop: should propagation method be delayed, if other propagators found reductions? (Default value = False)
        :param needscons: should the constraint handler be skipped, if no constraints are available? (Default value = True)
        :param proptiming: positions in the node solving loop where propagation method of constraint handlers should be executed (Default value = SCIP_PROPTIMING.BEFORELP)
        :param presoltiming: timing mask of the constraint handler's presolving method (Default value = SCIP_PRESOLTIMING.MEDIUM)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeConshdlr(self._scip, n, d, sepapriority, enfopriority, chckpriority, sepafreq, propfreq, eagerfreq,
                                              maxprerounds, delaysepa, delayprop, needscons, proptiming, presoltiming,
                                              PyConshdlrCopy, PyConsFree, PyConsInit, PyConsExit, PyConsInitpre, PyConsExitpre,
                                              PyConsInitsol, PyConsExitsol, PyConsDelete, PyConsTrans, PyConsInitlp, PyConsSepalp, PyConsSepasol,
                                              PyConsEnfolp, PyConsEnforelax, PyConsEnfops, PyConsCheck, PyConsProp, PyConsPresol, PyConsResprop, PyConsLock,
                                              PyConsActive, PyConsDeactive, PyConsEnable, PyConsDisable, PyConsDelvars, PyConsPrint, PyConsCopy,
                                              PyConsParse, PyConsGetvars, PyConsGetnvars, PyConsGetdivebdchgs,
                                              <SCIP_CONSHDLRDATA*>conshdlr))
        conshdlr.model = <Model>weakref.proxy(self)
        conshdlr.name = name
        Py_INCREF(conshdlr)

    def createCons(self, Conshdlr conshdlr, name, initial=True, separate=True, enforce=True, check=True, propagate=True,
                   local=False, modifiable=False, dynamic=False, removable=False, stickingatnode=False):
        """Create a constraint of a custom constraint handler

        :param Conshdlr conshdlr: constraint handler
        :param name: name of constraint
        :param initial:  (Default value = True)
        :param separate:  (Default value = True)
        :param enforce:  (Default value = True)
        :param check:  (Default value = True)
        :param propagate:  (Default value = True)
        :param local:  (Default value = False)
        :param modifiable:  (Default value = False)
        :param dynamic:  (Default value = False)
        :param removable:  (Default value = False)
        :param stickingatnode:  (Default value = False)

        """

        n = str_conversion(name)
        cdef SCIP_CONSHDLR* scip_conshdlr
        scip_conshdlr = SCIPfindConshdlr(self._scip, str_conversion(conshdlr.name))
        constraint = Constraint()
        PY_SCIP_CALL(SCIPcreateCons(self._scip, &(constraint.cons), n, scip_conshdlr, <SCIP_CONSDATA*>constraint,
                                initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode))
        return constraint

    def includePresol(self, Presol presol, name, desc, priority, maxrounds, timing=SCIP_PRESOLTIMING_FAST):
        """Include a presolver

        :param Presol presol: presolver
        :param name: name of presolver
        :param desc: description of presolver
        :param priority: priority of the presolver (>= 0: before, < 0: after constraint handlers)
        :param maxrounds: maximal number of presolving rounds the presolver participates in (-1: no limit)
        :param timing: timing mask of presolver (Default value = SCIP_PRESOLTIMING_FAST)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludePresol(self._scip, n, d, priority, maxrounds, timing, PyPresolCopy, PyPresolFree, PyPresolInit,
                                            PyPresolExit, PyPresolInitpre, PyPresolExitpre, PyPresolExec, <SCIP_PRESOLDATA*>presol))
        presol.model = <Model>weakref.proxy(self)
        Py_INCREF(presol)

    def includeSepa(self, Sepa sepa, name, desc, priority=0, freq=10, maxbounddist=1.0, usessubscip=False, delay=False):
        """Include a separator

        :param Sepa sepa: separator
        :param name: name of separator
        :param desc: description of separator
        :param priority: priority of separator (>= 0: before, < 0: after constraint handlers)
        :param freq: frequency for calling separator
        :param maxbounddist: maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying separation
        :param usessubscip: does the separator use a secondary SCIP instance? (Default value = False)
        :param delay: should separator be delayed, if other separators found cuts? (Default value = False)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeSepa(self._scip, n, d, priority, freq, maxbounddist, usessubscip, delay, PySepaCopy, PySepaFree,
                                          PySepaInit, PySepaExit, PySepaInitsol, PySepaExitsol, PySepaExeclp, PySepaExecsol, <SCIP_SEPADATA*>sepa))
        sepa.model = <Model>weakref.proxy(self)
        sepa.name = name
        Py_INCREF(sepa)

    def includeProp(self, Prop prop, name, desc, presolpriority, presolmaxrounds,
                    proptiming, presoltiming=SCIP_PRESOLTIMING_FAST, priority=1, freq=1, delay=True):
        """Include a propagator.

        :param Prop prop: propagator
        :param name: name of propagator
        :param desc: description of propagator
        :param presolpriority: presolving priority of the propgator (>= 0: before, < 0: after constraint handlers)
        :param presolmaxrounds: maximal number of presolving rounds the propagator participates in (-1: no limit)
        :param proptiming: positions class Modelin the node solving loop where propagation method of constraint handlers should be executed
        :param presoltiming: timing mask of the constraint handler's presolving method (Default value = SCIP_PRESOLTIMING_FAST)
        :param priority: priority of the propagator (Default value = 1)
        :param freq: frequency for calling propagator (Default value = 1)
        :param delay: should propagator be delayed if other propagators have found reductions? (Default value = True)

        """
        n = str_conversion(name)
        d = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeProp(self._scip, n, d,
                                          priority, freq, delay,
                                          proptiming, presolpriority, presolmaxrounds,
                                          presoltiming, PyPropCopy, PyPropFree, PyPropInit, PyPropExit,
                                          PyPropInitpre, PyPropExitpre, PyPropInitsol, PyPropExitsol,
                                          PyPropPresol, PyPropExec, PyPropResProp,
                                          <SCIP_PROPDATA*> prop))
        prop.model = <Model>weakref.proxy(self)
        Py_INCREF(prop)

    def includeHeur(self, Heur heur, name, desc, dispchar, priority=10000, freq=1, freqofs=0,
                    maxdepth=-1, timingmask=SCIP_HEURTIMING_BEFORENODE, usessubscip=False):
        """Include a primal heuristic.

        :param Heur heur: heuristic
        :param name: name of heuristic
        :param desc: description of heuristic
        :param dispchar: display character of heuristic
        :param priority: priority of the heuristic (Default value = 10000)
        :param freq: frequency for calling heuristic (Default value = 1)
        :param freqofs: frequency offset for calling heuristic (Default value = 0)
        :param maxdepth: maximal depth level to call heuristic at (Default value = -1)
        :param timingmask: positions in the node solving loop where heuristic should be executed (Default value = SCIP_HEURTIMING_BEFORENODE)
        :param usessubscip: does the heuristic use a secondary SCIP instance? (Default value = False)

        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        dis = ord(str_conversion(dispchar))
        PY_SCIP_CALL(SCIPincludeHeur(self._scip, nam, des, dis,
                                          priority, freq, freqofs,
                                          maxdepth, timingmask, usessubscip,
                                          PyHeurCopy, PyHeurFree, PyHeurInit, PyHeurExit,
                                          PyHeurInitsol, PyHeurExitsol, PyHeurExec,
                                          <SCIP_HEURDATA*> heur))
        heur.model = <Model>weakref.proxy(self)
        heur.name = name
        Py_INCREF(heur)

    def includeBranchrule(self, Branchrule branchrule, name, desc, priority, maxdepth, maxbounddist):
        """Include a branching rule.

        :param Branchrule branchrule: branching rule
        :param name: name of branching rule
        :param desc: description of branching rule
        :param priority: priority of branching rule
        :param maxdepth: maximal depth level up to which this branching rule should be used (or -1)
        :param maxbounddist: maximal relative distance from current node's dual bound to primal bound compared to best node's dual bound for applying branching rule (0.0: only on current best node, 1.0: on all nodes)

        """
        nam = str_conversion(name)
        des = str_conversion(desc)
        PY_SCIP_CALL(SCIPincludeBranchrule(self._scip, nam, des,
                                          priority, maxdepth, maxbounddist,
                                          PyBranchruleCopy, PyBranchruleFree, PyBranchruleInit, PyBranchruleExit,
                                          PyBranchruleInitsol, PyBranchruleExitsol, PyBranchruleExeclp, PyBranchruleExecext,
                                          PyBranchruleExecps, <SCIP_BRANCHRULEDATA*> branchrule))
        branchrule.model = <Model>weakref.proxy(self)
        Py_INCREF(branchrule)

    def getLPBranchCands(self):
        """Get the branching candidates from the LP relaxation.
        """
        cdef SCIP_VAR** lpcands
        cdef SCIP_Real* lpcandssol
        cdef SCIP_Real* lpcandsfrac
        cdef int nlpcands
        cdef int npriolpcands
        cdef int nfracimplvars
        PY_SCIP_CALL(SCIPgetLPBranchCands(self._scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, &npriolpcands, &nfracimplvars))

        variables = [Variable.create(lpcands[i]) for i in range(nlpcands)]

        return variables

    def getChildren(self):
        cdef SCIP_NODE** children
        cdef int nchildren
        PY_SCIP_CALL(SCIPgetChildren(self._scip, &children, &nchildren))
        return [Node.create(children[i]) for i in range(nchildren)]

    def branchVar(self, Variable var):
        PY_SCIP_CALL(SCIPbranchVar(self._scip, var.var, NULL, NULL, NULL))

    def reliabilityPseudocostBranching(self):
        cdef SCIP_VAR** lpcands
        cdef SCIP_Real* lpcandssol
        cdef SCIP_Real* lpcandsfrac
        cdef int nlpcands
        cdef int npriolpcands
        cdef int nfracimplvars
        cdef SCIP_RESULT result
        PY_SCIP_CALL(SCIPgetLPBranchCands(self._scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, &npriolpcands, &nfracimplvars))
        PY_SCIP_CALL(SCIPexecRelpscostBranching(self._scip, lpcands, lpcandssol, lpcandsfrac, nlpcands, True, &result))
        return result

    # Solution functions

    def createSol(self, Heur heur = None):
        """Create a new primal solution.

        :param Heur heur: heuristic that found the solution (Default value = None)

        """
        cdef SCIP_HEUR* _heur

        if isinstance(heur, Heur):
            n = str_conversion(heur.name)
            _heur = SCIPfindHeur(self._scip, n)
        else:
            _heur = NULL
        solution = Solution()
        PY_SCIP_CALL(SCIPcreateSol(self._scip, &solution.sol, _heur))
        return solution

    def printBestSol(self, write_zeros=False):
        """Prints the best feasible primal solution."""
        PY_SCIP_CALL(SCIPprintBestSol(self._scip, NULL, write_zeros))

    def printSol(self, Solution solution, write_zeros=False):
      """Print the given primal solution.

      Keyword arguments:
      solution -- solution to print
      write_zeros -- include variables that are set to zero
      """
      PY_SCIP_CALL(SCIPprintSol(self._scip, solution.sol, NULL, write_zeros))

    def writeBestSol(self, filename="origprob.sol", write_zeros=False):
        """Write the best feasible primal solution to a file.

        Keyword arguments:
        filename -- name of the output file
        write_zeros -- include variables that are set to zero
        """
        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintBestSol(self._scip, cfile, write_zeros))

    def writeSol(self, Solution solution, filename="origprob.sol", write_zeros=False):
        """Write the given primal solution to a file.

        Keyword arguments:
        solution -- solution to write
        filename -- name of the output file
        write_zeros -- include variables that are set to zero
        """
        # use this doubled opening pattern to ensure that IOErrors are
        #   triggered early and in Python not in C,Cython or SCIP.
        with open(filename, "w") as f:
            cfile = fdopen(f.fileno(), "w")
            PY_SCIP_CALL(SCIPprintSol(self._scip, solution.sol, cfile, write_zeros))

    # perhaps this should not be included as it implements duplicated functionality
    #   (as does it's namesake in SCIP)
    def readSol(self, filename):
        """Reads a given solution file, problem has to be transformed in advance.

        Keyword arguments:
        filename -- name of the input file
        """
        fn = str_conversion(filename)
        PY_SCIP_CALL(SCIPreadSol(self._scip, fn))

    def readSolFile(self, filename):
        """Reads a given solution file.

        Solution is created but not added to storage/the model.
        Use 'addSol' OR 'trySol' to add it.

        Keyword arguments:
        filename -- name of the input file
        """
        cdef SCIP_Bool partial
        cdef SCIP_Bool error
        cdef SCIP_Bool stored
        cdef Solution solution

        fn = str_conversion(filename)
        solution = self.createSol()
        PY_SCIP_CALL(SCIPreadSolFile(self._scip, fn, solution.sol, False, &partial, &error))
        if error:
            raise Exception("SCIP: reading solution from file failed!")

        return solution

    def setSolVal(self, Solution solution, Variable var, val):
        """Set a variable in a solution.

        :param Solution solution: solution to be modified
        :param Variable var: variable in the solution
        :param val: value of the specified variable

        """
        cdef SCIP_SOL* _sol
        _sol = <SCIP_SOL*>solution.sol
        PY_SCIP_CALL(SCIPsetSolVal(self._scip, _sol, var.var, val))

    def trySol(self, Solution solution, printreason=True, completely=False, checkbounds=True, checkintegrality=True, checklprows=True, free=True):
        """Check given primal solution for feasibility and try to add it to the storage.

        :param Solution solution: solution to store
        :param printreason: should all reasons of violations be printed? (Default value = True)
        :param completely: should all violation be checked? (Default value = False)
        :param checkbounds: should the bounds of the variables be checked? (Default value = True)
        :param checkintegrality: has integrality to be checked? (Default value = True)
        :param checklprows: have current LP rows (both local and global) to be checked? (Default value = True)
        :param free: should solution be freed? (Default value = True)

        """
        cdef SCIP_Bool stored
        if free:
            PY_SCIP_CALL(SCIPtrySolFree(self._scip, &solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        else:
            PY_SCIP_CALL(SCIPtrySol(self._scip, solution.sol, printreason, completely, checkbounds, checkintegrality, checklprows, &stored))
        return stored

    def addSol(self, Solution solution, free=True):
        """Try to add a solution to the storage.

        :param Solution solution: solution to store
        :param free: should solution be freed afterwards? (Default value = True)

        """
        cdef SCIP_Bool stored
        if free:
            PY_SCIP_CALL(SCIPaddSolFree(self._scip, &solution.sol, &stored))
        else:
            PY_SCIP_CALL(SCIPaddSol(self._scip, solution.sol, &stored))
        return stored

    def freeSol(self, Solution solution):
        """Free given solution

        :param Solution solution: solution to be freed

        """
        PY_SCIP_CALL(SCIPfreeSol(self._scip, &solution.sol))

    def getSols(self):
        """Retrieve list of all feasible primal solutions stored in the solution storage."""
        cdef SCIP_SOL** _sols
        cdef SCIP_SOL* _sol
        _sols = SCIPgetSols(self._scip)
        nsols = SCIPgetNSols(self._scip)
        sols = []

        for i in range(nsols):
            sols.append(Solution.create(_sols[i]))

        return sols

    def getBestSol(self):
        """Retrieve currently best known feasible primal solution."""
        self._bestSol = Solution.create(SCIPgetBestSol(self._scip))
        return self._bestSol

    def getSolObjVal(self, Solution sol, original=True):
        """Retrieve the objective value of the solution.

        :param Solution sol: solution
        :param original: objective value in original space (Default value = True)

        """
        if sol == None:
            sol = Solution.create(NULL)
        if original:
            objval = SCIPgetSolOrigObj(self._scip, sol.sol)
        else:
            objval = SCIPgetSolTransObj(self._scip, sol.sol)
        return objval

    def getObjVal(self, original=True):
        """Retrieve the objective value of value of best solution.
        Can only be called after solving is completed.

        :param original: objective value in original space (Default value = True)

        """
        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")
        return self.getSolObjVal(self._bestSol, original)

    def getSolVal(self, Solution sol, Variable var):
        """Retrieve value of given variable in the given solution or in
        the LP/pseudo solution if sol == None

        :param Solution sol: solution
        :param Variable var: variable to query the value of

        """
        if sol == None:
            sol = Solution.create(NULL)
        return SCIPgetSolVal(self._scip, sol.sol, var.var)

    def getVal(self, Variable var):
        """Retrieve the value of the best known solution.
        Can only be called after solving is completed.

        :param Variable var: variable to query the value of

        """
        if not self.getStage() >= SCIP_STAGE_SOLVING:
            raise Warning("method cannot be called before problem is solved")
        return self.getSolVal(self._bestSol, var)

    def getPrimalbound(self):
        """Retrieve the best primal bound."""
        return SCIPgetPrimalbound(self._scip)

    def getDualbound(self):
        """Retrieve the best dual bound."""
        return SCIPgetDualbound(self._scip)

    def getDualboundRoot(self):
        """Retrieve the best root dual bound."""
        return SCIPgetDualboundRoot(self._scip)

    def writeName(self, Variable var):
        """Write the name of the variable to the std out.

        :param Variable var: variable

        """
        PY_SCIP_CALL(SCIPwriteVarName(self._scip, NULL, var.var, False))

    def getStage(self):
        """Retrieve current SCIP stage"""
        return SCIPgetStage(self._scip)

    def getStatus(self):
        """Retrieve solution status."""
        cdef SCIP_STATUS stat = SCIPgetStatus(self._scip)
        if stat == SCIP_STATUS_OPTIMAL:
            return "optimal"
        elif stat == SCIP_STATUS_TIMELIMIT:
            return "timelimit"
        elif stat == SCIP_STATUS_INFEASIBLE:
            return "infeasible"
        elif stat == SCIP_STATUS_UNBOUNDED:
            return "unbounded"
        else:
            return "unknown"

    def getObjectiveSense(self):
        """Retrieve objective sense."""
        cdef SCIP_OBJSENSE sense = SCIPgetObjsense(self._scip)
        if sense == SCIP_OBJSENSE_MAXIMIZE:
            return "maximize"
        elif sense == SCIP_OBJSENSE_MINIMIZE:
            return "minimize"
        else:
            return "unknown"

    def catchEvent(self, eventtype, Eventhdlr eventhdlr):
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPcatchEvent(self._scip, eventtype, _eventhdlr, NULL, NULL))

    def dropEvent(self, eventtype, Eventhdlr eventhdlr):
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPdropEvent(self._scip, eventtype, _eventhdlr, NULL, -1))

    def catchVarEvent(self, Variable var, eventtype, Eventhdlr eventhdlr):
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPcatchVarEvent(self._scip, var.var, eventtype, _eventhdlr, NULL, NULL))

    def dropVarEvent(self, Variable var, eventtype, Eventhdlr eventhdlr):
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPdropVarEvent(self._scip, var.var, eventtype, _eventhdlr, NULL, -1))

    def catchRowEvent(self, Row row, eventtype, Eventhdlr eventhdlr):
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPcatchRowEvent(self._scip, row.row, eventtype, _eventhdlr, NULL, NULL))

    def dropRowEvent(self, Row row, eventtype, Eventhdlr eventhdlr):
        cdef SCIP_EVENTHDLR* _eventhdlr
        if isinstance(eventhdlr, Eventhdlr):
            n = str_conversion(eventhdlr.name)
            _eventhdlr = SCIPfindEventhdlr(self._scip, n)
        else:
            raise Warning("event handler not found")
        PY_SCIP_CALL(SCIPdropRowEvent(self._scip, row.row, eventtype, _eventhdlr, NULL, -1))

    # Statistic Methods

    def printStatistics(self):
        """Print statistics."""
        PY_SCIP_CALL(SCIPprintStatistics(self._scip, NULL))

    def writeStatistics(self, filename="origprob.stats"):
      """Write statistics to a file.

      Keyword arguments:
      filename -- name of the output file
      """
      # use this doubled opening pattern to ensure that IOErrors are
      #   triggered early and in Python not in C,Cython or SCIP.
      with open(filename, "w") as f:
          cfile = fdopen(f.fileno(), "w")
          PY_SCIP_CALL(SCIPprintStatistics(self._scip, cfile))

    def getNLPs(self):
        return SCIPgetNLPs(self._scip)

    # Verbosity Methods

    def hideOutput(self, quiet = True):
        """Hide the output.

        :param quiet: hide output? (Default value = True)

        """
        SCIPsetMessagehdlrQuiet(self._scip, quiet)

    # Parameter Methods

    def setBoolParam(self, name, value):
        """Set a boolean-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetBoolParam(self._scip, n, value))

    def setIntParam(self, name, value):
        """Set an int-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetIntParam(self._scip, n, value))

    def setLongintParam(self, name, value):
        """Set a long-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetLongintParam(self._scip, n, value))

    def setRealParam(self, name, value):
        """Set a real-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetRealParam(self._scip, n, value))

    def setCharParam(self, name, value):
        """Set a char-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPsetCharParam(self._scip, n, value))

    def setStringParam(self, name, value):
        """Set a string-valued parameter.

        :param name: name of parameter
        :param value: value of parameter

        """
        n = str_conversion(name)
        v = str_conversion(value)
        PY_SCIP_CALL(SCIPsetStringParam(self._scip, n, v))

    def setParam(self, name, value):
        """Set a parameter with value in int, bool, real, long, char or str.

        :param name: name of parameter
        :param value: value of parameter
        """
        cdef SCIP_PARAM* param

        n = str_conversion(name)
        param = SCIPgetParam(self._scip, n)

        if param == NULL:
            raise KeyError("Not a valid parameter name")

        paramtype =  SCIPparamGetType(param)

        if paramtype == SCIP_PARAMTYPE_BOOL:
            PY_SCIP_CALL(SCIPsetBoolParam(self._scip, n, value))
        elif paramtype == SCIP_PARAMTYPE_INT:
            PY_SCIP_CALL(SCIPsetIntParam(self._scip, n, value))
        elif paramtype == SCIP_PARAMTYPE_LONGINT:
            PY_SCIP_CALL(SCIPsetLongintParam(self._scip, n, value))
        elif paramtype == SCIP_PARAMTYPE_REAL:
            PY_SCIP_CALL(SCIPsetRealParam(self._scip, n, value))
        elif paramtype == SCIP_PARAMTYPE_CHAR:
            PY_SCIP_CALL(SCIPsetCharParam(self._scip, n, value))
        elif paramtype == SCIP_PARAMTYPE_STRING:
            PY_SCIP_CALL(SCIPsetStringParam(self._scip, n, value))


    def getParam(self, name):
        """Get the value of a parameter of type
        int, bool, real, long, char or str.

        :param name: name of parameter
        """
        cdef SCIP_PARAM* param

        n = str_conversion(name)
        param = SCIPgetParam(self._scip, n)

        if param == NULL:
            raise KeyError("Not a valid parameter name")

        paramtype =  SCIPparamGetType(param)

        if paramtype == SCIP_PARAMTYPE_BOOL:
            return SCIPparamGetBool(param)
        elif paramtype == SCIP_PARAMTYPE_INT:
            return SCIPparamGetInt(param)
        elif paramtype == SCIP_PARAMTYPE_LONGINT:
            return SCIPparamGetLongint(param)
        elif paramtype == SCIP_PARAMTYPE_REAL:
            return SCIPparamGetReal(param)
        elif paramtype == SCIP_PARAMTYPE_CHAR:
            return SCIPparamGetChar(param)
        elif paramtype == SCIP_PARAMTYPE_STRING:
            return SCIPparamGetString(param)


    def readParams(self, file):
        """Read an external parameter file.

        :param file: file to be read

        """
        absfile = str_conversion(abspath(file))
        PY_SCIP_CALL(SCIPreadParams(self._scip, absfile))

    def writeParams(self, filename='param.set', comments = True, onlychanged = True):
        """Write parameter settings to an external file.

        :param filename: file to be written (Default value = 'param.set')
        :param comments: write parameter descriptions as comments? (Default value = True)
        :param onlychanged: write only modified parameters (Default value = True)

        """
        fn = str_conversion(filename)
        PY_SCIP_CALL(SCIPwriteParams(self._scip, fn, comments, onlychanged))
        print('wrote parameter settings to file ' + filename)

    def resetParam(self, name):
        """Reset parameter setting to its default value

        :param name: parameter to reset

        """
        n = str_conversion(name)
        PY_SCIP_CALL(SCIPresetParam(self._scip, n))

    def resetParams(self):
        """Reset parameter settings to their default values"""
        PY_SCIP_CALL(SCIPresetParams(self._scip))

    def setEmphasis(self, paraemphasis, quiet = True):
        """Set emphasis settings

        :param paraemphasis: emphasis to set
        :param quiet: hide output? (Default value = True)

        """
        PY_SCIP_CALL(SCIPsetEmphasis(self._scip, paraemphasis, quiet))

    def readProblem(self, file, extension = None):
        """Read a problem instance from an external file.

        :param file: file to be read
        :param extension: specify file extension/type (Default value = None)

        """
        absfile = str_conversion(abspath(file))
        if extension is None:
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, NULL))
        else:
            extension = str_conversion(extension)
            PY_SCIP_CALL(SCIPreadProb(self._scip, absfile, extension))

    # Counting functions

    def count(self):
        """Counts the number of feasible points of problem."""
        PY_SCIP_CALL(SCIPcount(self._scip))

    def getNCountedSols(self):
        """Get number of feasible solution."""
        cdef SCIP_Bool valid
        cdef SCIP_Longint nsols

        nsols = SCIPgetNCountedSols(self._scip, &valid)
        if not valid:
            print('total number of solutions found is not valid!')
        return nsols

    def setParamsCountsols(self):
        """Sets SCIP parameters such that a valid counting process is possible."""
        PY_SCIP_CALL(SCIPsetParamsCountsols(self._scip))

    def freeReoptSolve(self):
        """Frees all solution process data and prepares for reoptimization"""
        PY_SCIP_CALL(SCIPfreeReoptSolve(self._scip))

    def chgReoptObjective(self, coeffs, sense = 'minimize'):
        """Establish the objective function as a linear expression.

        :param coeffs: the coefficients
        :param sense: the objective sense (Default value = 'minimize')

        """

        cdef SCIP_OBJSENSE objsense

        if sense == "minimize":
            objsense = SCIP_OBJSENSE_MINIMIZE
        elif sense == "maximize":
            objsense = SCIP_OBJSENSE_MAXIMIZE
        else:
            raise Warning("unrecognized optimization sense: %s" % sense)

        assert isinstance(coeffs, Expr)

        if coeffs.degree() > 1:
            raise ValueError("Nonlinear objective functions are not supported!")
        if coeffs[CONST] != 0.0:
            raise ValueError("Constant offsets in objective are not supported!")

        cdef SCIP_VAR** _vars
        cdef int _nvars
        _vars = SCIPgetOrigVars(self._scip)
        _nvars = SCIPgetNOrigVars(self._scip)
        _coeffs = <SCIP_Real*> malloc(_nvars * sizeof(SCIP_Real))

        for i in range(_nvars):
            _coeffs[i] = 0.0

        for term, coef in coeffs.terms.items():
            # avoid CONST term of Expr
            if term != CONST:
                assert len(term) == 1
                var = <Variable>term[0]
                for i in range(_nvars):
                    if _vars[i] == var.var:
                        _coeffs[i] = coef

        PY_SCIP_CALL(SCIPchgReoptObjective(self._scip, objsense, _vars, &_coeffs[0], _nvars))

        free(_coeffs)


    ###########################################
    #       branch-search-trees methods       #
    ###########################################

    def getNLPIterations(self):
        """Get number of LP iterations executed so far."""
        return SCIPgetNLPIterations(self._scip)

    def getMaxDepth(self):
        """Get max depth."""
        return SCIPgetMaxDepth(self._scip)

    def getNNodesLeft(self):
        """Return the number of nodes left (leaves + children + siblings)."""
        return SCIPgetNNodesLeft(self._scip)

    def getNDiscreteVars(self, transformed=False):
        """Get number of binary + integer variables.
        
        :param transformed: bool, get transformed variables instead of original (Default value = False)
        """
        return SCIPgetNBinVars(self._scip) + SCIPgetNIntVars(self._scip)

    def getRootNode(self):
        """Retrieve root node."""
        return Node.create(SCIPgetRootNode(self._scip))

    # lower and upper bounds refer to the transformed problem
    def getLowerboundRoot(self):
        """Return the lower bound at the root node."""
        return SCIPgetLowerboundRoot(self._scip)

    def getLowerbound(self):
        """Return the global lower (dual) bound in transformed problem."""
        return SCIPgetLowerbound(self._scip)

    def getUpperbound(self):
        """Return the global upper (primal) bound in transformed problem."""
        return SCIPgetUpperbound(self._scip)

    def getTransGap(self):
        """Return gets current gap |(upperbound - lowerbound)/min(|upperbound|,|lowerbound|)| in transformed problem."""
        return SCIPgetTransGap(self._scip)

    # dual and primal bounds refer to the original problem
    def getDualboundRoot(self):
        """Return the dual bound of the root node for the original problem."""
        return SCIPgetDualboundRoot(self._scip)

    def getDualbound(self):
        """Return the global dual bound."""
        return SCIPgetDualbound(self._scip)

    def getPrimalbound(self):
        """Return the global primal bound (objective value of best solution or user objective limit) for the original problem."""
        return SCIPgetPrimalbound(self._scip)

    def getLPObjval(self):
        """Return the objective value of current LP (which is the sum of column and loose objective value)."""
        return SCIPgetLPObjval(self._scip)

    def getNLeaves(self):
        """Return the total number of processed leaf nodes."""
        return SCIPgetNObjlimLeaves(self._scip) + SCIPgetNFeasibleLeaves(self._scip) + SCIPgetNInfeasibleLeaves(self._scip)

    def getNObjlimLeaves(self):
        """Return the number of processed leaf nodes that hit LP objective limit."""
        return SCIPgetNObjlimLeaves(self._scip)

    def getNInfeasibleLeaves(self):
        """Return the number of infeasible leaf nodes processed."""
        return SCIPgetNInfeasibleLeaves(self._scip)

    def getNCreatedNodes(self):
        """Return total number of nodes created."""
        return self._scip.stat.ncreatednodes

    def getNCreatedNodesRun(self):
        """Return number of nodes created in current run."""
        return self._scip.stat.ncreatednodesrun
        
    def getPrimalDualIntegral(self):
        """Return the primal dual integral."""
        return self._scip.stat.primaldualintegral
    
    def getNumLPBranchCands(self):
        """Get the number of branching candidates from the LP relaxation."""
        cdef SCIP_VAR** lpcands
        cdef SCIP_Real* lpcandssol
        cdef SCIP_Real* lpcandsfrac
        cdef int nlpcands
        cdef int npriolpcands
        cdef int nfracimplvars
        PY_SCIP_CALL(SCIPgetLPBranchCands(self._scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, &npriolpcands, &nfracimplvars))
        return nlpcands

    def relPosition(self, node_bound, ub, lb):
        """Relative position of node_bound with respect to global upper and lower bounds (or other commensurable quantities).
        
        :param node_bound: float, LP bound at node
        :param ub: float, global upper bound
        :param lb: float, global lower bound
        """
        if SCIPisInfinity(self._scip, ub):
            return 0.
        else:
            return np.abs(ub - node_bound) / np.abs(ub -lb)
    
    # static
    def relDistance(self, x, y):
        """Relative distance between x and y."""
        if x*y<0:
            return 0.
        else:
            return np.abs(x-y) / np.max([np.abs(x), np.abs(y), 1e-10])
    
    # static
    def gNormMax(self, x):
        """Normalizer function g(x) (cf. T. Achterberg PhD Thesis)."""
        gx = x/(x+1.)
        return np.max([gx, 0.1])
        
    # static
    def getVarScore(self, varscore, avgscore):
        """Returns a branching variable score with respect to the average one.
        (cf. relpscost formula as in calcScore https://scip.zib.de/doc-6.0.0/html/branch__relpscost_8c_source.php#l00347)
        """
        maxavg = np.max([avgscore, 0.1])
        return 1 - (1 / (1 + varscore / maxavg))
        
    def getFairNNodes(self, brancher_name):
        """Returns the number of fair nodes for the specified branching rule.
        
        :param brancher_name: str, name of the branching rule 
        """
        brancher = SCIPfindBranchrule(self._scip, brancher_name)
        fair = SCIPgetNNodes(self._scip) + 2*SCIPbranchruleGetNCutoffs(brancher) + 2*SCIPbranchruleGetNDomredsFound(brancher)
        return fair    
      
    # Hand-crafted Input Parameterizations
    # The parameterization Tree_t of the search tree is given by concatenation of getNodeState and getMIPState 
    # Candidate variables representation C_t is defined in getCandsState
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getNodeState(self, node_dim):
        """Get Node state representation.
        
        :param node_dim: int, dimensionality of node state representation (8).
        """
        node_state = np.empty(node_dim, dtype=np.double)
        cdef double[::1] node_state_view = node_state # C-view contiguous

        cdef SCIP_NODE* node = SCIPgetCurrentNode(self._scip)
        domchg = SCIPnodeGetDomchg(node)
        nboundchgs = SCIPdomchgGetNBoundchgs(domchg)

        if node == SCIPgetRootNode(self._scip):
            isRoot = True
        else:
            isRoot = False

        # depth and position
        if isRoot:
            node_state_view[0:2] = 0.
        else:
            node_state_view[0] = float(SCIPnodeGetDepth(node)) / SCIPgetMaxDepth(self._scip)
            node_state_view[1] = float(SCIPgetPlungeDepth(self._scip)) / SCIPnodeGetDepth(node)
                 
        # LP objective
        node_state_view[2] = self.relDistance(SCIPgetLowerbound(self._scip), SCIPgetLPObjval(self._scip))
        node_state_view[3] = self.relDistance(SCIPgetLowerboundRoot(self._scip), SCIPgetLPObjval(self._scip))
        
        # LP bound
        if SCIPisInfinity(self._scip, SCIPgetUpperbound(self._scip)):
            node_state_view[4:6] = 0.
        else:
            node_state_view[4] = self.relDistance(SCIPgetUpperbound(self._scip), SCIPgetLPObjval(self._scip))
            node_state_view[5] = self.relPosition(node_bound=SCIPgetLPObjval(self._scip), ub=SCIPgetUpperbound(self._scip), lb=SCIPgetLowerbound(self._scip)) 
        
        # candidate set and bound changes
        node_state_view[6] = float(len(self.getLPBranchCands())) / self.getNDiscreteVars()
        node_state_view[7] = float(nboundchgs) / SCIPgetNVars(self._scip)
        
        return node_state

        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getMIPState(self, mip_dim):
        """Get MIP state representation.
        
        :param mip_dim: int, dimensionality of MIP state representation (53).
        """
        
        mip_state = np.empty(mip_dim, dtype=np.double)
        cdef double[::1] mip_state_view = mip_state  # C-view contiguous

        cdef SCIP_NODE** leaves
        cdef SCIP_NODE** children
        cdef SCIP_NODE** siblings
        cdef int nleaves
        cdef int nchildren
        cdef int nsiblings
        PY_SCIP_CALL(SCIPgetOpenNodesData(self._scip, &leaves, &children, &siblings, &nleaves, &nchildren, &nsiblings))

        if SCIPgetCurrentNode(self._scip) == SCIPgetRootNode(self._scip):
            isRoot = True
        else:
            isRoot = False
        
        if isRoot:
            # nodes and leaves
            mip_state_view[:8] = 0.
            # depth and backtracks
            mip_state_view[8:12] = 0.
            # LP iterations
            mip_state_view[12:15] = 0.
            if SCIPgetNLPs(self._scip) != 0:
                mip_state_view[15] = float(SCIPgetNNodeLPs(self._scip)) / SCIPgetNLPs(self._scip)
            else:
                mip_state_view[15] = 0.
        
        else:
            # nodes and leaves
            all_leaves = SCIPgetNObjlimLeaves(self._scip) + SCIPgetNFeasibleLeaves(self._scip) + SCIPgetNInfeasibleLeaves(self._scip)
            if all_leaves == 0:
                mip_state_view[:3] = 0.
            else:
                mip_state_view[0] = float(SCIPgetNObjlimLeaves(self._scip)) / all_leaves 
                mip_state_view[1] = float(SCIPgetNInfeasibleLeaves(self._scip)) / all_leaves 
                mip_state_view[2] = float(SCIPgetNFeasibleLeaves(self._scip)) / all_leaves 
            mip_state_view[3] = float((SCIPgetNInfeasibleLeaves(self._scip) + 1.0) / (SCIPgetNObjlimLeaves(self._scip) + 1.0)) # dynamic factor formula
            mip_state_view[4] = float(SCIPgetNNodesLeft(self._scip)) / SCIPgetNNodes(self._scip) 
            mip_state_view[5] = float(all_leaves) / SCIPgetNNodes(self._scip)
            mip_state_view[6] = float(self._scip.stat.ninternalnodes) / SCIPgetNNodes(self._scip) 
            mip_state_view[7] = float(SCIPgetNNodes(self._scip)) / self._scip.stat.ncreatednodes 

            # depth and backtracks
            mip_state_view[8] = float(self._scip.stat.nactivatednodes) / SCIPgetNNodes(self._scip) 
            mip_state_view[9] = float(self._scip.stat.ndeactivatednodes) / SCIPgetNNodes(self._scip)
            mip_state_view[10] = float(SCIPgetPlungeDepth(self._scip)) / SCIPgetMaxDepth(self._scip)
            mip_state_view[11] = float(SCIPgetNBacktracks(self._scip)) / SCIPgetNNodes(self._scip)

            # LP iterations
            mip_state_view[12] = np.log(float(SCIPgetNLPIterations(self._scip)) / SCIPgetNNodes(self._scip))
            mip_state_view[13] = np.log(float(SCIPgetNLPs(self._scip)) / SCIPgetNNodes(self._scip)) 
            mip_state_view[14] = float(SCIPgetNNodes(self._scip)) / SCIPgetNLPs(self._scip) 
            mip_state_view[15] = float(SCIPgetNNodeLPs(self._scip)) / SCIPgetNLPs(self._scip)
            
        # gap
        if self._scip.stat.primaldualintegral == 0:
            mip_state_view[16] = 0.
        else:
            mip_state_view[16] = np.log(self._scip.stat.primaldualintegral)

        if SCIPisInfinity(self._scip, SCIPgetGap(self._scip)) or SCIPisInfinity(self._scip, self._scip.stat.lastsolgap) or SCIPisInfinity(self._scip, self._scip.stat.firstsolgap):
            mip_state_view[17:20] = 0.
        else:    
            mip_state_view[17] = SCIPgetGap(self._scip) / self._scip.stat.lastsolgap
            mip_state_view[18] = SCIPgetGap(self._scip) / self._scip.stat.firstsolgap
            mip_state_view[19] = self._scip.stat.lastsolgap / self._scip.stat.firstsolgap
        
        # bounds and solutions
        mip_state_view[20] = self.relDistance(SCIPgetLowerboundRoot(self._scip), SCIPgetLowerbound(self._scip))
        mip_state_view[21] = self.relDistance(SCIPgetLowerboundRoot(self._scip), SCIPgetAvgLowerbound(self._scip))
        if SCIPisInfinity(self._scip, SCIPgetUpperbound(self._scip)):
            mip_state_view[22] = 0.
        else:
            mip_state_view[22] = self.relDistance(SCIPgetUpperbound(self._scip), SCIPgetLowerbound(self._scip)) # zero until UB is available

        mip_state_view[23] = float(SCIPisPrimalboundSol(self._scip))
        if isRoot:
            mip_state_view[24] = 0.
        else:
            mip_state_view[24] = float(self._scip.stat.nnodesbeforefirst) / SCIPgetNNodes(self._scip) # negative until first solution found
        
        # scores (cf. branching scores)
        mip_state_view[25] = self.gNormMax(SCIPgetAvgConflictScore(self._scip))
        mip_state_view[26] = self.gNormMax(SCIPgetAvgConflictlengthScore(self._scip))
        mip_state_view[27] = self.gNormMax(SCIPgetAvgInferenceScore(self._scip))
        mip_state_view[28] = self.gNormMax(SCIPgetAvgCutoffScore(self._scip))
        mip_state_view[29] = self.gNormMax(SCIPgetAvgPseudocostScore(self._scip))
        
        mip_state_view[30] = self.gNormMax(SCIPgetAvgCutoffs(self._scip, SCIP_BRANCHDIR_UPWARDS)) 
        mip_state_view[31] = self.gNormMax(SCIPgetAvgCutoffs(self._scip, SCIP_BRANCHDIR_DOWNWARDS))
        mip_state_view[32] = self.gNormMax(SCIPgetAvgInferences(self._scip, SCIP_BRANCHDIR_UPWARDS))
        mip_state_view[33] = self.gNormMax(SCIPgetAvgInferences(self._scip, SCIP_BRANCHDIR_DOWNWARDS))
        mip_state_view[34] = self.gNormMax(SCIPgetPseudocostVariance(self._scip, SCIP_BRANCHDIR_UPWARDS, 1))
        mip_state_view[35] = self.gNormMax(SCIPgetPseudocostVariance(self._scip, SCIP_BRANCHDIR_DOWNWARDS, 1))
        mip_state_view[36] = self.gNormMax(SCIPgetNConflictConssApplied(self._scip)) 
        
        # open nodes
        open_lowerbounds = np.empty([nleaves + nchildren + nsiblings], dtype = np.double)
        cdef double[::1] open_lowerbounds_view = open_lowerbounds  # C-view contiguous
        open_depths = np.empty([nleaves + nchildren + nsiblings], dtype = np.double)
        cdef double[::1] open_depths_view = open_depths  # C-view contiguous

        if nleaves + nchildren + nsiblings != 0:
            for i in range(nleaves):
                open_lowerbounds_view[i] = leaves[i].lowerbound
                open_depths_view[i] = leaves[i].depth
            for i in range(nchildren):
                open_lowerbounds_view[nleaves+i] = children[i].lowerbound
                open_depths_view[nleaves+i] = children[i].depth
            for i in range(nsiblings):
                open_lowerbounds_view[nleaves+nchildren+i] = siblings[i].lowerbound
                open_depths_view[nleaves+nchildren+i] = siblings[i].depth
        
            # bounds
            mip_state_view[37] = float(len(np.argwhere(open_lowerbounds == np.min(open_lowerbounds)))) / len(open_lowerbounds) 
            mip_state_view[38] = float(len(np.argwhere(open_lowerbounds == np.max(open_lowerbounds)))) / len(open_lowerbounds)
            mip_state_view[39] = self.relDistance(SCIPgetLowerbound(self._scip), np.max(open_lowerbounds))
            mip_state_view[40] = self.relDistance(np.min(open_lowerbounds), np.max(open_lowerbounds)) 
            if SCIPisInfinity(self._scip, SCIPgetUpperbound(self._scip)):
                mip_state_view[41:46] = 0.
            else:
                mip_state_view[41] = self.relDistance(np.min(open_lowerbounds), SCIPgetUpperbound(self._scip))
                mip_state_view[42] = self.relDistance(np.max(open_lowerbounds), SCIPgetUpperbound(self._scip))
                mip_state_view[43] = self.relPosition(node_bound=np.mean(open_lowerbounds), ub=SCIPgetUpperbound(self._scip), lb=SCIPgetLowerbound(self._scip)) 
                mip_state_view[44] = self.relPosition(node_bound=np.min(open_lowerbounds), ub=SCIPgetUpperbound(self._scip), lb=SCIPgetLowerbound(self._scip)) 
                mip_state_view[45] = self.relPosition(node_bound=np.max(open_lowerbounds), ub=SCIPgetUpperbound(self._scip), lb=SCIPgetLowerbound(self._scip)) 
            
            lb_q1 = np.quantile(open_lowerbounds, 0.25)
            lb_q3 = np.quantile(open_lowerbounds, 0.75)
            mip_state_view[46] = self.relDistance(lb_q1, lb_q3) 
            if np.mean(open_lowerbounds) == 0:
                mip_state_view[47] = 0.
            else: 
                mip_state_view[47] = np.std(open_lowerbounds)/np.mean(open_lowerbounds) # coefficient of variation
            if lb_q1 + lb_q3 == 0:
                mip_state_view[48] = 0.
            else:
                mip_state_view[48] = (lb_q3 - lb_q1)/(lb_q3 + lb_q1) # quartile coefficient of dispersion
               
            # depths
            d_q1 = np.quantile(open_depths, 0.25)
            d_q3 = np.quantile(open_depths, 0.75)
            if SCIPgetMaxDepth(self._scip) == 0: 
                mip_state_view[49] = 0.
            else:
                mip_state_view[49] = float(np.mean(open_depths)) / SCIPgetMaxDepth(self._scip)
            mip_state_view[50] = self.relDistance(d_q1, d_q3)
            if np.mean(open_depths) == 0:
                mip_state_view[51] = 0.
            else: 
                mip_state_view[51] = np.std(open_depths)/np.mean(open_depths) # coefficient of variation
            if d_q1 + d_q3 == 0:
                mip_state_view[52] = 0.
            else:
                mip_state_view[52] = (d_q3 - d_q1)/(d_q3 + d_q1) # quartile coefficient of dispersion
        else:
            mip_state_view[37:] = 0.
        
        return mip_state
        
 
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getCandsState(self, var_dim, branch_count):
        """Get candidate variables state representation.
        
        :param var_dim: int, dimensionality of candidate variables state representation (25).
        :param branch_count: int, counter of number of branchings performed.
        """
        # get candidate variables
        cdef SCIP_VAR** lpcands
        cdef SCIP_Real* lpcandssol
        cdef SCIP_Real* lpcandsfrac
        cdef SCIP_STAT* stat = self._scip.stat
        cdef int nlpcands
        cdef int npriolpcands
        cdef int nfracimplvars
        PY_SCIP_CALL(SCIPgetLPBranchCands(self._scip, &lpcands, &lpcandssol, &lpcandsfrac, &nlpcands, &npriolpcands, &nfracimplvars))

        cands_state_mat = np.empty((nlpcands, var_dim), dtype=np.double)
        cdef double[:, ::1] cands_state_mat_view = cands_state_mat  # C-view contiguous
        cdef size_t i = 0
    
        for i in range(nlpcands):
            # general solution
            cands_state_mat_view[i][0] = lpcandsfrac[i] # same as SCIPvarGetLPSol
            cands_state_mat_view[i][1] = SCIPvarGetAvgSol(lpcands[i]) # across feasible primal solutions
            # branching depth
            if SCIPgetMaxDepth(self._scip) == 0:
                cands_state_mat_view[i][2:4] = 0.
            else:
                cands_state_mat_view[i][2] = 1 - (SCIPvarGetAvgBranchdepthCurrentRun(lpcands[i], SCIP_BRANCHDIR_UPWARDS) / SCIPgetMaxDepth(self._scip)) # could be <0
                cands_state_mat_view[i][3] = 1 - (SCIPvarGetAvgBranchdepthCurrentRun(lpcands[i], SCIP_BRANCHDIR_DOWNWARDS) / SCIPgetMaxDepth(self._scip))
            
            # branching scores (cf. relpscost formula https://scip.zib.de/doc-6.0.0/html/branch__relpscost_8c_source.php#l00524 )
            cands_state_mat_view[i][4] = self.getVarScore(SCIPgetVarConflictScore(self._scip, lpcands[i]), SCIPgetAvgConflictScore(self._scip))
            cands_state_mat_view[i][5] = self.getVarScore(SCIPgetVarConflictlengthScore(self._scip, lpcands[i]), SCIPgetAvgConflictlengthScore(self._scip))
            cands_state_mat_view[i][6] = self.getVarScore(SCIPgetVarAvgInferenceScore(self._scip, lpcands[i]), SCIPgetAvgInferenceScore(self._scip))
            cands_state_mat_view[i][7] = self.getVarScore(SCIPgetVarAvgCutoffScore(self._scip, lpcands[i]), SCIPgetAvgCutoffScore(self._scip))
            cands_state_mat_view[i][8] = self.getVarScore(SCIPgetVarPseudocostScore(self._scip, lpcands[i], lpcandssol[i]), SCIPgetAvgPseudocostScore(self._scip))
            
            # pc stats
            if SCIPgetPseudocostCount(self._scip, SCIP_BRANCHDIR_UPWARDS, 1) == 0:
                cands_state_mat_view[i][9] = 0.
            else:
                cands_state_mat_view[i][9] = SCIPgetVarPseudocostCountCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_UPWARDS) / SCIPgetPseudocostCount(self._scip, SCIP_BRANCHDIR_UPWARDS, 1)
            if SCIPgetPseudocostCount(self._scip, SCIP_BRANCHDIR_DOWNWARDS, 1) == 0:
                cands_state_mat_view[i][10] = 0.
            else: 
                cands_state_mat_view[i][10] = SCIPgetVarPseudocostCountCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_DOWNWARDS) / SCIPgetPseudocostCount(self._scip, SCIP_BRANCHDIR_DOWNWARDS, 1)
            
            if SCIPvarGetNBranchingsCurrentRun(lpcands[i], SCIP_BRANCHDIR_UPWARDS) == 0:
                cands_state_mat_view[i][11] = 0.
            else:
                cands_state_mat_view[i][11] = SCIPgetVarPseudocostCountCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_UPWARDS) / SCIPvarGetNBranchingsCurrentRun(lpcands[i], SCIP_BRANCHDIR_UPWARDS)
            if SCIPvarGetNBranchingsCurrentRun(lpcands[i], SCIP_BRANCHDIR_DOWNWARDS) == 0:
                cands_state_mat_view[i][12] = 0.
            else:
                cands_state_mat_view[i][12] = SCIPgetVarPseudocostCountCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_DOWNWARDS) / SCIPvarGetNBranchingsCurrentRun(lpcands[i], SCIP_BRANCHDIR_DOWNWARDS) 
            
            if branch_count == 0:
                cands_state_mat_view[i][13:15] = 0.
            else:
                cands_state_mat_view[i][13] = SCIPgetVarPseudocostCountCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_UPWARDS) / branch_count 
                cands_state_mat_view[i][14] = SCIPgetVarPseudocostCountCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_DOWNWARDS) / branch_count 
            # other roles/effects
            # implications
            cands_state_mat_view[i][15] = SCIPvarGetNImpls(lpcands[i], 0) 
            cands_state_mat_view[i][16] = SCIPvarGetNImpls(lpcands[i], 1) 
            # cliques
            if SCIPgetNCliques(self._scip) == 0:
                cands_state_mat_view[i][17:19] = 0.
            else:
                cands_state_mat_view[i][17] = SCIPvarGetNCliques(lpcands[i], 0) / SCIPgetNCliques(self._scip) 
                cands_state_mat_view[i][18] = SCIPvarGetNCliques(lpcands[i], 1) / SCIPgetNCliques(self._scip) 
            # cutoffs
            cands_state_mat_view[i][19] = self.gNormMax(SCIPgetVarAvgCutoffsCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_UPWARDS))
            cands_state_mat_view[i][20] = self.gNormMax(SCIPgetVarAvgCutoffsCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_DOWNWARDS))
            # conflict length
            cands_state_mat_view[i][21] = self.gNormMax(SCIPgetVarAvgConflictlengthCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_UPWARDS))
            cands_state_mat_view[i][22] = self.gNormMax(SCIPgetVarAvgConflictlengthCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_DOWNWARDS))
            # inferences
            cands_state_mat_view[i][23] = self.gNormMax(SCIPgetVarAvgInferencesCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_UPWARDS))
            cands_state_mat_view[i][24] = self.gNormMax(SCIPgetVarAvgInferencesCurrentRun(self._scip, lpcands[i], SCIP_BRANCHDIR_DOWNWARDS))
            
        return [Variable.create(lpcands[i]) for i in range(nlpcands)], [SCIPcolGetLPPos(SCIPvarGetCol(lpcands[i])) for i in range(nlpcands)], cands_state_mat


    def executeBranchRule(self, str name, allowaddcons):
        cdef SCIP_BRANCHRULE*  branchrule
        cdef SCIP_RESULT result
        branchrule = SCIPfindBranchrule(self._scip, name.encode("UTF-8"))
        if branchrule == NULL:
            print("Error, branching rule not found!")
            return PY_SCIP_RESULT.DIDNOTFIND
        else:
            branchrule.branchexeclp(self._scip, branchrule, allowaddcons, &result)
            return result

    ###########################################

# debugging memory management
def is_memory_freed():
    return BMSgetMemoryUsed() == 0

def print_memory_in_use():
    BMScheckEmptyMemory()
