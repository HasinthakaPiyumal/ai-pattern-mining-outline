# Cluster 3

def generate_images(prompt: str, num_predictions: int):
    tokenized_prompt = tokenize_prompt(prompt)
    seed = random.randint(0, 2 ** 32 - 1)
    key = jax.random.PRNGKey(seed)
    images = []
    for i in range(max(1, num_predictions // jax.device_count())):
        key, subkey = jax.random.split(key)
        encoded_images = p_generate(tokenized_prompt, shard_prng_key(subkey), params, gen_top_k, gen_top_p, temperature, cond_scale)
        encoded_images = encoded_images.sequences[..., 1:]
        decoded_images = p_decode(encoded_images, vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        for img in decoded_images:
            images.append(Image.fromarray(np.asarray(img * 255, dtype=np.uint8)))
    return images

def tokenize_prompt(prompt: str):
    tokenized_prompt = processor([prompt])
    return replicate(tokenized_prompt)

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
    return model.generate(**tokenized_prompt, prng_key=key, params=params, top_k=top_k, top_p=top_p, temperature=temperature, condition_scale=condition_scale)

