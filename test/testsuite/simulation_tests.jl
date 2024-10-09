using Test
using ActionModels

@testset "simulate actions" begin

    agent = premade_agent("binary_rescorla_wagner_softmax", verbose = false)

    inputs = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

    actions = give_inputs!(agent, inputs)

    @test length(actions) == length(inputs)
end
