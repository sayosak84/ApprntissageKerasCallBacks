"?D
BHostIDLE"IDLE1㥛? <?@A㥛? <?@a*???ԫ??i*???ԫ???Unknown
^HostGatherV2"GatherV2(1^?Iǁ@9^?Iǁ@A^?Iǁ@I^?Iǁ@a??8S?^??i?+é????Unknown
oHost_FusedMatMul"sequential/dense/Relu(1{?G??|@9{?G??|@A{?G??|@I{?G??|@a#F??KϷ?i??2Ʃ{???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?Q???|@9?Q???|@A?Q???|@I?Q???|@a???2Y???i??4n???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?t??y@9?t??y@A?t??y@I?t??y@a?J?E8u??i!?F?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1??/??w@9??/??w@A??/??w@I??/??w@a?[?ϼ??iZ)??u????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1}?5^?Mw@9}?5^?Mw@A}?5^?Mw@I}?5^?Mw@ah?d????i\V"?t????Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1X9??X@9X9??X@AX9??X@IX9??X@aV?n????i??-ڛ???Unknown
?	HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1?????]M@9?????]M@A?????]M@I?????]M@a?:?B??i/?b?????Unknown
}
HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1/?$??K@9/?$??K@A/?$??K@I/?$??K@a'??(6??i???:Y???Unknown
tHost_FusedMatMul"sequential/dense_1/BiasAdd(1`??"ۙA@9`??"ۙA@A`??"ۙA@I`??"ۙA@a???}?i??h7????Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1?t?>@9?t?>@A?t?>@I?t?>@a???f8?x?i?ic??????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?5^?I?;@9?5^?I?;@A?5^?I?;@I?5^?I?;@a?s???v?iP??????Unknown
oHostSoftmax"sequential/dense_1/Softmax(1??x?&1:@9??x?&1:@A??x?&1:@I??x?&1:@a
???âu?i??F????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1)\???H9@9)\???H9@A)\???H9@I)\???H9@a?????t?ik7?3?G???Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1????x?2@9????x?2@A????x?2@I????x?2@agA??	o?i?N???f???Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1sh??|_6@9sh??|_6@A/?$?2@I/?$?2@au^ ?n?iƬ??J????Unknown
qHostCast"sequential/dropout/dropout/Cast(1?MbXy2@9?MbXy2@A?MbXy2@I?MbXy2@a??!?O?n?i???4У???Unknown
iHostWriteSummary"WriteSummary(1??(\?0@9??(\?0@A??(\?0@I??(\?0@aS??2sj?iޔ?gC????Unknown?
rHostDataset"Iterator::Root::ParallelMapV2(1/?$??'@9/?$??'@A/?$??'@I/?$??'@a;???c?i??q??????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1????S#@9????S#@A????S#@I????S#@aH???_?i>?]??????Unknown
cHostDataset"Iterator::Root(1+?y5@9+?y5@A?&1?#@I?&1?#@a?$2%?_?i???Y?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1%??C?"@9%??C?"@A%??C?"@I%??C?"@a]U?ڦ_?iYd?:???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1ףp=
!@9ףp=
!@Aףp=
!@Iףp=
!@a?ZB1?;\?i??|?X???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1?C?l?{@9?C?l?{@A?C?l?{@I?C?l?{@a*Žnb.Y?ii??????Unknown
oHostMul"sequential/dropout/dropout/Mul(1?C?l??@9?C?l??@A?C?l??@I?C?l??@a???	?W?iU?8?~'???Unknown
?HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1/?$?Y@9/?$?Y@A?p=
?#@I?p=
?#@a瓭?ėU?iђ?J2???Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1?A`??"@9?A`??"@A?A`??"@I?A`??"@a3??(??U?i?%?=???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aN? ?)zU?iL6?#?G???Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_2(1bX9??@9bX9??@AbX9??@IbX9??@aZ?G_|LU?iگayR???Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1?V-@9?V-@A?V-@I?V-@a?5????T?i3L?V?\???Unknown
[ HostAddV2"Adam/add(1;?O??n@9;?O??n@A;?O??n@I;?O??n@a
[S?isQ#??f???Unknown
e!Host
LogicalAnd"
LogicalAnd(1??"???@9??"???@A??"???@I??"???@a?x]?WQ?i?R?8o???Unknown?
}"HostMul",gradient_tape/sequential/dropout/dropout/Mul(11?Zd@91?Zd@A1?Zd@I1?Zd@a??^@?P?i???w???Unknown
l#HostIteratorGetNext"IteratorGetNext(1NbX94@9NbX94@ANbX94@INbX94@a??\?~?P?i?H?A????Unknown
`$HostGatherV2"
GatherV2_1(1??ʡE?@9??ʡE?@A??ʡE?@I??ʡE?@a?$J??K?i}QH?????Unknown
w%HostDataset""Iterator::Root::ParallelMapV2::Zip(1`??"?)`@9`??"?)`@A'1?Z@I'1?Z@a??&E??I?i$eb?]????Unknown
?&HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?ʡE??@9?ʡE????A?ʡE??@I?ʡE????ad`??Y?I?i<J????Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1j?t?@9j?t?@Aj?t?@Ij?t?@a?)??j?H?ię??????Unknown
?(HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1?Zd;@9?Zd;@A?Zd;@I?Zd;@a.??v?%H?i?x?]?????Unknown
t)HostReadVariableOp"Adam/Cast/ReadVariableOp(1??~j?t@9??~j?t@A??~j?t@I??~j?t@a!??[?F?i?n?ᩥ???Unknown
?*HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1?5^?I@9?5^?I@A?5^?I@I?5^?I@aq?>??WF?i?~???????Unknown
?+HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1o??ʡ
@9o??ʡ
@Ao??ʡ
@Io??ʡ
@aZQ?~??E?i?%lǿ????Unknown
T,HostMul"Mul(1???K7?
@9???K7?
@A???K7?
@I???K7?
@a|?;v??E?i??	?:????Unknown
Z-HostArgMax"ArgMax(1/?$??	@9/?$??	@A/?$??	@I/?$??	@a6{~81"E?iA?W4?????Unknown
V.HostSum"Sum_2(1??Q??@9??Q??@A??Q??@I??Q??@aC?P?0AD?ieh???????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1?l????@9?l????@A?l????@I?l????@aY?b?C?i?k.Y?????Unknown
o0HostReadVariableOp"Adam/ReadVariableOp(1??"??~@9??"??~@A??"??~@I??"??~@aW?J¥hC?ir???_????Unknown
?1HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1???S??@9???S??@A???S??@I???S??@aX!??R?B?i??G?????Unknown
\2HostArgMax"ArgMax_1(1??MbX@9??MbX@A??MbX@I??MbX@a?o?]	uB?i-??????Unknown
[3HostPow"
Adam/Pow_1(1?I+?@9?I+?@A?I+?@I?I+?@a?䏖z?A?i?D8????Unknown
Y4HostPow"Adam/Pow(1?C?l??@9?C?l??@A?C?l??@I?C?l??@a"s#q]??i}???????Unknown
v5HostCast"$sparse_categorical_crossentropy/Cast(1F????x@9F????x@AF????x@IF????x@a6?mw?>?i5??g?????Unknown
X6HostEqual"Equal(1??K7?A@9??K7?A@A??K7?A@I??K7?A@a?3?C-?<?i;)?h????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_4(1??(\????9??(\????A??(\????I??(\????a\???7<:?iW#4?????Unknown
?8HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1??(\????9??(\????A??(\????I??(\????a???$H?8?i?"??????Unknown
]9HostCast"Adam/Cast_1(1?~j?t???9?~j?t???A?~j?t???I?~j?t???aI??X)?0?ib?M??????Unknown
b:HostDivNoNan"div_no_nan_1(1V-?????9V-?????AV-?????IV-?????a????š-?i?(?޼????Unknown
?;HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1^?I+??9^?I+??A^?I+??I^?I+??a?8?^?\,?i???????Unknown
v<HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1
ףp=
??9
ףp=
??A
ףp=
??I
ףp=
??a??ݝ?&,?i??+E????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a???$+?i?y??????Unknown
?>HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?$??C??9?$??C??A?$??C??I?$??C??a6?RT?)?i???ғ????Unknown
??HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1??|?5^??9??|?5^??A??|?5^??I??|?5^??a?I??)?i??i0%????Unknown
X@HostCast"Cast_2(1?E??????9?E??????A?E??????I?E??????a??@)
?'?i?7A?????Unknown
yAHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1/?$????9/?$????A/?$????I/?$????a??Y? ?&?i1??????Unknown
vBHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1R???Q??9R???Q??AR???Q??IR???Q??aq???$?i??qAP????Unknown
`CHostDivNoNan"
div_no_nan(1Zd;?O???9Zd;?O???AZd;?O???IZd;?O???a.*?\}t#?i۽G??????Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_3(1?(\?????9?(\?????A?(\?????I?(\?????aºTz]P!?i'c??????Unknown
uEHostReadVariableOp"div_no_nan/ReadVariableOp(1??C?l???9??C?l???A??C?l???I??C?l???a6?.q ?iF??????Unknown
aFHostIdentity"Identity(17?A`????97?A`????A7?A`????I7?A`????a?????i???O?????Unknown?
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1/?$????9/?$????A/?$????I/?$????a??Y? ??ioǭ?R????Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1=
ףp=??9=
ףp=??A=
ףp=??I=
ףp=??ay"G???i      ???Unknown*?C
^HostGatherV2"GatherV2(1^?Iǁ@9^?Iǁ@A^?Iǁ@I^?Iǁ@a??5?4??i??5?4???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1{?G??|@9{?G??|@A{?G??|@I{?G??|@ab?6^????i3??????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?Q???|@9?Q???|@A?Q???|@I?Q???|@a??Df???i`?#?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?t??y@9?t??y@A?t??y@I?t??y@a?u+?8??ij#? ????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1??/??w@9??/??w@A??/??w@I??/??w@a????׽?i?aRK???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1}?5^?Mw@9}?5^?Mw@A}?5^?Mw@I}?5^?Mw@a???d???iT??a????Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1X9??X@9X9??X@AX9??X@IX9??X@aQ?2? ??i??b????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1?????]M@9?????]M@A?????]M@I?????]M@aI????V??i?EV?r???Unknown
}	HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1/?$??K@9/?$??K@A/?$??K@I/?$??K@a?йk??i^Nֽu????Unknown
t
Host_FusedMatMul"sequential/dense_1/BiasAdd(1`??"ۙA@9`??"ۙA@A`??"ۙA@I`??"ۙA@ae;M????iL?3FdU???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1?t?>@9?t?>@A?t?>@I?t?>@aj&??ɂ?i?c?J?????Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1?5^?I?;@9?5^?I?;@A?5^?I?;@I?5^?I?;@a????G??i??F?????Unknown
oHostSoftmax"sequential/dense_1/Softmax(1??x?&1:@9??x?&1:@A??x?&1:@I??x?&1:@af??;[??i??u4'???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1)\???H9@9)\???H9@A)\???H9@I)\???H9@a????w??iv??#Af???Unknown?
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1????x?2@9????x?2@A????x?2@I????x?2@a??]?vw?i??Q?.????Unknown
?HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1sh??|_6@9sh??|_6@A/?$?2@I/?$?2@a&~?KZw?i????g????Unknown
qHostCast"sequential/dropout/dropout/Cast(1?MbXy2@9?MbXy2@A?MbXy2@I?MbXy2@a{??w?i??c
?????Unknown
iHostWriteSummary"WriteSummary(1??(\?0@9??(\?0@A??(\?0@I??(\?0@a?Ѡ???s?is?????Unknown?
rHostDataset"Iterator::Root::ParallelMapV2(1/?$??'@9/?$??'@A/?$??'@I/?$??'@a?????m?i???rO7???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1????S#@9????S#@A????S#@I????S#@a??Y4?#h?i6V?BsO???Unknown
cHostDataset"Iterator::Root(1+?y5@9+?y5@A?&1?#@I?&1?#@a??u5??g?i?Rg???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1%??C?"@9%??C?"@A%??C?"@I%??C?"@a!?Syg?i9VW?~???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1ףp=
!@9ףp=
!@Aףp=
!@Iףp=
!@a????Xe?i?i%V#????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1?C?l?{@9?C?l?{@A?C?l?{@I?C?l?{@aL诞a	c?i?ķ,????Unknown
oHostMul"sequential/dropout/dropout/Mul(1?C?l??@9?C?l??@A?C?l??@I?C?l??@aL!?Զya?iݞ?n?????Unknown
?HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1/?$?Y@9/?$?Y@A?p=
?#@I?p=
?#@an??p?R`?ip?	Z?????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1?A`??"@9?A`??"@A?A`??"@I?A`??"@a'HӼGR`?i?hơK????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?H#Љ<`?i??+?????Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_2(1bX9??@9bX9??@AbX9??@IbX9??@a?g??`?ii=g-?????Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1?V-@9?V-@A?V-@I?V-@a6ly??q_?i?B?Z	???Unknown
[HostAddV2"Adam/add(1;?O??n@9;?O??n@A;?O??n@I;?O??n@a=A D]?i??R?????Unknown
e Host
LogicalAnd"
LogicalAnd(1??"???@9??"???@A??"???@I??"???@a?T!9Z?iEڌ%???Unknown?
}!HostMul",gradient_tape/sequential/dropout/dropout/Mul(11?Zd@91?Zd@A1?Zd@I1?Zd@a???xY?i!??1???Unknown
l"HostIteratorGetNext"IteratorGetNext(1NbX94@9NbX94@ANbX94@INbX94@an?h?;Y?iQ|U?s>???Unknown
`#HostGatherV2"
GatherV2_1(1??ʡE?@9??ʡE?@A??ʡE?@I??ʡE?@a??c?T?i?~=3?H???Unknown
w$HostDataset""Iterator::Root::ParallelMapV2::Zip(1`??"?)`@9`??"?)`@A'1?Z@I'1?Z@a??P%?S?i1??E?R???Unknown
?%HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?ʡE??@9?ʡE????A?ʡE??@I?ʡE????aO??2TS?i?4__W\???Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_2(1j?t?@9j?t?@Aj?t?@Ij?t?@a??J??R?i|???e???Unknown
?'HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1?Zd;@9?Zd;@A?Zd;@I?Zd;@a(߈_:AR?i?@?D?n???Unknown
t(HostReadVariableOp"Adam/Cast/ReadVariableOp(1??~j?t@9??~j?t@A??~j?t@I??~j?t@a?PR?M%Q?i?i??iw???Unknown
?)HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1?5^?I@9?5^?I@A?5^?I@I?5^?I@a?RB,?P?iۊ??????Unknown
?*HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1o??ʡ
@9o??ʡ
@Ao??ʡ
@Io??ʡ
@aH?????P?i??B?,????Unknown
T+HostMul"Mul(1???K7?
@9???K7?
@A???K7?
@I???K7?
@a???:?P?i?O?u????Unknown
Z,HostArgMax"ArgMax(1/?$??	@9/?$??	@A/?$??	@I/?$??	@aZh?C?O?il???r????Unknown
V-HostSum"Sum_2(1??Q??@9??Q??@A??Q??@I??Q??@a??@?ޟN?i?E[?????Unknown
t.HostAssignAddVariableOp"AssignAddVariableOp(1?l????@9?l????@A?l????@I?l????@a?J?B??M?i??kŔ????Unknown
o/HostReadVariableOp"Adam/ReadVariableOp(1??"??~@9??"??~@A??"??~@I??"??~@am??vXM?iX???????Unknown
?0HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1???S??@9???S??@A???S??@I???S??@a\?5ITIL?i? '8?????Unknown
\1HostArgMax"ArgMax_1(1??MbX@9??MbX@A??MbX@I??MbX@a?AT!?K?i1|@?????Unknown
[2HostPow"
Adam/Pow_1(1?I+?@9?I+?@A?I+?@I?I+?@a?<:?J?iR1?????Unknown
Y3HostPow"Adam/Pow(1?C?l??@9?C?l??@A?C?l??@I?C?l??@a2?,ӵG?iZփ?????Unknown
v4HostCast"$sparse_categorical_crossentropy/Cast(1F????x@9F????x@AF????x@IF????x@a??W	G?i??b????Unknown
X5HostEqual"Equal(1??K7?A@9??K7?A@A??K7?A@I??K7?A@al?5S?E?iIf`?????Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_4(1??(\????9??(\????A??(\????I??(\????a??_?C?i??+??????Unknown
?7HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1??(\????9??(\????A??(\????I??(\????a|??A??B?i?*? `????Unknown
]8HostCast"Adam/Cast_1(1?~j?t???9?~j?t???A?~j?t???I?~j?t???aAaJײ9?iw?{?????Unknown
b9HostDivNoNan"div_no_nan_1(1V-?????9V-?????AV-?????IV-?????a??dt?f6?i??Sc????Unknown
?:HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1^?I+??9^?I+??A^?I+??I^?I+??aPt?0q5?i'??y????Unknown
v;HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1
ףp=
??9
ףp=
??A
ףp=
??I
ףp=
??a?@?6CH5?iO?"??????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff??9ffffff??Affffff??Iffffff??a?(?
?{4?i?d?I????Unknown
?=HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?$??C??9?$??C??A?$??C??I?$??C??a?>???3?i??޸?????Unknown
?>HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1??|?5^??9??|?5^??A??|?5^??I??|?5^??aPa?6??2?i???????Unknown
X?HostCast"Cast_2(1?E??????9?E??????A?E??????I?E??????a?w?H2?i?֥?Y????Unknown
y@HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1/?$????9/?$????A/?$????I/?$????aǹ?v?91?in?4??????Unknown
vAHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1R???Q??9R???Q??AR???Q??IR???Q??am?w??_.?i?u??f????Unknown
`BHostDivNoNan"
div_no_nan(1Zd;?O???9Zd;?O???AZd;?O???IZd;?O???a???W^j-?iR???=????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1?(\?????9?(\?????A?(\?????I?(\?????a??d?-*?i?;?s?????Unknown
uDHostReadVariableOp"div_no_nan/ReadVariableOp(1??C?l???9??C?l???A??C?l???I??C?l???a?R???(?i??v3n????Unknown
aEHostIdentity"Identity(17?A`????97?A`????A7?A`????I7?A`????a]?'?i?85?????Unknown?
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1/?$????9/?$????A/?$????I/?$????aǹ?v?9!?iry??????Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1=
ףp=??9=
ףp=??A=
ףp=??I=
ףp=??aO?h?b ?i      ???Unknown2CPU