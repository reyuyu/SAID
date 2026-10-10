Initial controller-development CPU run: 46 passed, 14 failed. No GPU precheck,
smoke, or formal training had started.

Eight failures requested an optional hns_graph in the native valid_count<2
fallback, which intentionally does not capture that graph. Those cases now
check the actual fallback loss, gradients, zero hierarchy term, and immutable
parameters directly.

Six failures mistakenly included weighted_hierarchy among the components
required to be unchanged. That value is the final macro-scaled term; its
measured 0.9 ratio was the desired behavior. The invariant checks now cover raw
hierarchy and both edge violations, while the final weighted term is explicitly
checked against 0.9 times baseline. The corresponding precheck invariant list
was corrected before GPU execution.

No production source, loss coefficient, training parameter, tolerance, data,
or historical artifact was changed to address these test implementation errors.
