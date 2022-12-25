expr<-t(as.matrix(expr))
stage<-stage[['flag']]
dataset <- wrap_expression(
  counts = expr,
  expression = expr
)

dataset <- add_prior_information(dataset, start_id = dataset$cell_ids[1])

# p1<-plot_dimred(
#   model,
#   expression_source = dataset$expression,
#   grouping = stage,
#   color_density = "grouping",
# )
#
# p2<-plot_dimred(
#   model,
#   color_density = "grouping",
#   grouping = dynwrap::group_onto_nearest_milestones(model)
#
# )

model <- infer_trajectory(dataset, method)
model <-dynwrap::add_root(model,dataset$cell_ids[1])
result<-calculate_pseudotime(model)
result<-sort(result)
