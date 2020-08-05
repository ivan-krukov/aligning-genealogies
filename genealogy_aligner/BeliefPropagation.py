def neighbors(node, arcs, except_node=None):

  neighbors = []
  for arc in arcs:
    if node in arc:
      for n in arc:
        if (n != node) and (n != except_node):
          if not n in neighbors:
            neighbors.append(n)
  return neighbors

def revise(Di, Dj, Xi, Xj, constrain_Xi_Xj):
  revised = False
  for x in Di:
    is_all_incompatible = True
    for y in Dj:
      is_all_incompatible = is_all_incompatible and ((x,y) in constrain_Xi_Xj)
    if is_all_incompatible:
      Di.remove(x)
      revised = True
  return revised

def arc_consistency(variables, domains, constrains):

  arcs = list(constrains.keys())

  if len(arcs) > 0:

    queue = [arcs[0]]

    while len(queue) > 0:
      Xi, Xj = queue.pop()
      if revise(domains[Xi], domains[Xj], Xi, Xj, constrains[(Xi,Xj)]):
        if len(domains[Xi]) == 0:
          return False
        neighbors_Xi = neighbors(Xi, arcs, Xj)
        for Xk in neighbors_Xi:
          if (Xk, Xi) in arcs:
            queue.append((Xk,Xi))
          else:
            queue.append((Xi,Xk))
  return True

def select_unassigned_variable(variables, domains, constrains, assignment, heuristic=None):
  non_assigned_vars = [v for v in variables if not v in assignment.keys()]
  if len(non_assigned_vars) > 0:
    return non_assigned_vars[0]
  else:
    return None

def order_domain_values(var, assignment, variables, domains, constrains):
  return domains[var]

def inference(var, value, variables, domains, constrains, assignment, inf_type = None):
  """
  Two different heuristics are implemented so far:
  1) Forward checking (FC)
  2) Maintaining Arc Consistency (MAC)
  """
  new_assignments = {}
  if inf_type == "FC":
    result = arc_consistency(variables, domains, constrains)
    if not result:
      new_assignments = "Failure"
  elif inf_type == "MAC":
    unassg_vars = []
    sub_constrains = {}
    for (var1,var2) in constrains.keys():
      if var in (var1,var2):
        other_var = None
        if var1 == var:
          other_var = var1
        else:
          other_var = var2
        if not other_var in assignment.keys():
          unassg_vars.append(other_var)
          sub_constrains[(var1,var2)] = constrains[(var1,var2)]
    # Call AC with the sub constrains and variables
    result = arc_consistency(unassg_vars, domains, sub_constrains)
    if not result:
      new_assignments = "Failure"


  return new_assignments




def is_consistent(var, value, assignment, constrains):

  is_consistent = True
  for var1 in assignment:
    if (var1,var) in constrains:
      if (assignment[var1],value) in constrains[(var1,var)]:
        is_consistent = False
    elif (var,var1) in constrains:
      if (value,assignment[var1]) in constrains[(var,var1)]:
        is_consistent = False
  return is_consistent





def backtrack(assignment, variables, domains, constrains, inf_type=None):

  if len(assignment) == len(variables):
    return assignment
  var = select_unassigned_variable(variables, domains, constrains, assignment)
  for value in order_domain_values(var, assignment, variables, domains, constrains):
    inference_assignments = None
    if is_consistent(var, value, assignment, constrains):
      assignment[var] = value
      _domain = domains.copy()
      inference_assignments = inference(var, value, variables, _domain, constrains, assignment, inf_type = inf_type)
      if inference_assignments != "Failure":
        for v in inference_assignments:
          assignment[v] = inference_assignments[v]
        result = backtrack(assignment, variables, domains, constrains)
        if result != "Failure":
          print (result)


if __name__ == "__main__":
  variables = [0, 1, 2, 3, 4]
  domains = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3], 3: [0, 1, 2, 3], 4: [0, 1, 2, 3]}
  constrains = {(4, 3): [(3, 2), (2, 1), (3, 1), (1, 1), (2, 1), (2, 2), (2, 1), (2, 1), (0, 1), (1, 3), (0, 3), (1, 2)], (1, 3): [(3, 3), (0, 2), (3, 0), (0, 0), (2, 3), (1, 3), (2, 1), (1, 3), (0, 2), (0, 1), (1, 2), (0, 3)], (2, 0): [(1, 2), (2, 1), (2, 0), (3, 0), (3, 0), (0, 0), (3, 2), (3, 2), (0, 2), (1, 1), (1, 3), (1, 3)]}
  backtrack ({}, variables, domains, constrains )