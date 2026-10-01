"""Check Cosmos checkpoint compatibility, token layout, masks, and gradients."""
import pytest

torch = pytest.importorskip("torch")

from world_models.models.cosmos_transformer import (
    CosmosTransformerConfig,
    ScratchCosmosTransformer,
    apply_rotary_embedding,
    patchify,
    unpatchify_output,
)


def small_config(**overrides):
    config = dict(
        in_channels=5,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=12,
        num_layers=2,
        mlp_ratio=2.0,
        text_embed_dim=20,
        adaln_lora_dim=8,
        max_size=(8,16,16),
        patch_size=(1,2,2),
        rope_scale=(1.0,3.0,3.0),
        concat_padding_mask=False,
        extra_pos_embed_type=None,
        use_crossattn_projection=True,
        crossattn_proj_in_channels=30,
        encoder_hidden_states_channels=20,
    )
    config.update(overrides)
    return config


def model_inputs(per_frame=False):
    torch.manual_seed(71)
    condition = torch.zeros(2,1,3,8,8)
    condition[:, :, 0] = 1
    return dict(
        hidden_states=torch.randn(2,4,3,8,8),
        timestep=torch.rand(2,1,3,1,1) if per_frame else torch.tensor([0.1,0.8]),
        encoder_hidden_states=torch.randn(2,5,30),
        condition_mask=condition,
        padding_mask=torch.zeros(1,1,16,16),
        attention_mask=torch.tensor([[True,True,False,False,False],[True,True,True,True,False]]),
        fps=16,
    )


@pytest.mark.parametrize("per_frame", [False, True])
@pytest.mark.parametrize("learned_positions", [False, True])
def test_full_forward_matches_diffusers_and_loads_all_keys(per_frame, learned_positions):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(42)
    reference = diffusers.CosmosTransformer3DModel(**small_config(
        extra_pos_embed_type="learnable" if learned_positions else None,
    )).eval()
    if learned_positions:
        # Released 2.5 disables this branch; nonzero values test the optional implementation.
        with torch.no_grad():
            for parameter in reference.learnable_pos_embed.parameters():
                parameter.normal_()
    model = ScratchCosmosTransformer.from_reference(reference)
    inputs = model_inputs(per_frame)
    with torch.no_grad():
        expected = reference(**inputs).sample
        actual = model(**inputs).sample
    assert set(reference.state_dict()) == set(model.state_dict())
    assert all(not parameter.is_meta for parameter in model.parameters())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_padding_resize_and_per_frame_conditioning_match_reference():
    pytest.importorskip("torchvision")
    diffusers = pytest.importorskip("diffusers")
    reference = diffusers.CosmosTransformer3DModel(**small_config(concat_padding_mask=True)).eval()
    model = ScratchCosmosTransformer.from_reference(reference)
    inputs = model_inputs(per_frame=True)
    inputs["padding_mask"][:, :, :8, :8] = 1
    with torch.no_grad():
        expected = reference(**inputs).sample
        actual = model(**inputs, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_masked_text_tokens_cannot_change_the_video_prediction():
    torch.manual_seed(13)
    model = ScratchCosmosTransformer(**small_config()).eval()
    inputs = model_inputs()
    with torch.no_grad():
        first = model(**inputs).sample
        changed_text = inputs["encoder_hidden_states"].clone()
        changed_text[~inputs["attention_mask"]] = 100 * torch.randn_like(changed_text[~inputs["attention_mask"]])
        inputs["encoder_hidden_states"] = changed_text
        second = model(**inputs).sample
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_conditioning_channel_and_text_receive_gradients():
    model = ScratchCosmosTransformer(**small_config())
    inputs = model_inputs(per_frame=True)
    inputs["encoder_hidden_states"].requires_grad_()
    inputs["condition_mask"].requires_grad_()
    model(**inputs).sample.square().mean().backward()
    for tensor in (inputs["encoder_hidden_states"], inputs["condition_mask"], model.transformer_blocks[0].attn1.to_q.weight):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_patch_layout_uses_channel_before_local_pixel_positions():
    video = torch.arange(32).view(1,2,1,4,4)
    patches = patchify(video, (1,2,2))
    assert patches.shape == (1,1,2,2,8)
    assert patches[0,0,0,0].tolist() == [0,1,4,5,16,17,20,21]
    assert patches[0,0,1,1].tolist() == [10,11,14,15,26,27,30,31]
    with pytest.raises(ValueError, match="not divisible"):
        patchify(torch.zeros(1,2,1,3,4), (1,2,2))


def test_output_projection_uses_pixel_before_channel_order():
    # One patch with four pixels, each carrying two output channels.
    projected = torch.tensor([[[10,20,11,21,12,22,13,23]]])
    decoded = unpatchify_output(projected, (1,1,1), (1,2,2))
    assert decoded[0,0,0].tolist() == [[10,11],[12,13]]
    assert decoded[0,1,0].tolist() == [[20,21],[22,23]]


def test_rotary_embedding_preserves_norm_and_uses_video_fps():
    model = ScratchCosmosTransformer(**small_config())
    x = torch.zeros(1,4,3,8,8)
    rotary_24 = model.rope(x, fps=24)
    rotary_12 = model.rope(x, fps=12)
    query = torch.randn(1,2,48,12)
    rotated = apply_rotary_embedding(query, rotary_24)
    torch.testing.assert_close(rotated.norm(dim=-1), query.norm(dim=-1))
    # Frame zero has the same positions at either fps; later frames differ.
    torch.testing.assert_close(rotary_24[0][:16], rotary_12[0][:16])
    assert not torch.equal(rotary_24[0][16:], rotary_12[0][16:])


def test_config_uses_released_channels_and_rejects_unsupported_features():
    config = CosmosTransformerConfig()
    assert config.hidden_size == 2048
    assert config.in_channels == 17
    assert config.rope_scale == (1.0,3.0,3.0)
    with pytest.raises(ValueError, match="ControlNet"):
        CosmosTransformerConfig(controlnet_block_every_n=7)
    with pytest.raises(ValueError, match="Unsupported configuration"):
        ScratchCosmosTransformer.from_config({**small_config(), "unknown_option": True})


def test_input_errors_explain_conditioning_contract():
    model = ScratchCosmosTransformer(**small_config(concat_padding_mask=True))
    inputs = model_inputs()
    with pytest.raises(ValueError, match="including condition_mask"):
        model(**{**inputs, "condition_mask": None})
    with pytest.raises(ValueError, match="padding_mask must"):
        model(**{**inputs, "padding_mask": None})
    with pytest.raises(ValueError, match="boolean attention mask"):
        model(**{**inputs, "attention_mask": inputs["attention_mask"].long()})
