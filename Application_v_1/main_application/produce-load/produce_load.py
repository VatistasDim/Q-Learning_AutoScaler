import aiohttp
import asyncio

ApplicationTrainUrl = 'http://application:8501/train'
#ApplicationTestUrl = 'http://application:8501/test'
cooldown = 5

async def make_request(session, url):
    try:
        response = await session.get(url)
        print(f"Response for {url}: {response.status}")
    except Exception as e:
        print(f"An error occurred for {url}: {e}")

async def make_requests():
    print("Making requests...")
    async with aiohttp.ClientSession() as session:
        # Run multiple requests concurrently
        tasks = [
            make_request(session, ApplicationTrainUrl),
            #make_request(session, ApplicationTestUrl),
        ]
        await asyncio.gather(*tasks)

async def main():
    while True:
        try:
            await make_requests()
            print(f"Cooldown for {cooldown} seconds")
            await asyncio.sleep(cooldown)
        except Exception as e:
            print(f"An error occurred: {e}, retry in {cooldown} sec.")
            await asyncio.sleep(cooldown)

if __name__ == "__main__":
    asyncio.run(main())
