{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "31477f8e-dd0f-454f-97e4-2e855656885d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Environment working!\n"
     ]
    }
   ],
   "source": [
    "# test_rl.py\n",
    "import gymnasium as gym\n",
    "\n",
    "env = gym.make(\"CartPole-v1\")\n",
    "\n",
    "obs, _ = env.reset()\n",
    "\n",
    "for _ in range(100):\n",
    "    action = env.action_space.sample()\n",
    "    obs, reward, done, truncated, info = env.step(action)\n",
    "\n",
    "    if done or truncated:\n",
    "        obs, _ = env.reset()\n",
    "\n",
    "print(\"Environment working!\")\n",
    "env.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "71d6056e-226f-427b-bae6-91ed01b3a895",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'test_rl' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mNameError\u001b[39m                                 Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[7]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m \u001b[43mtest_rl\u001b[49m.py\n",
      "\u001b[31mNameError\u001b[39m: name 'test_rl' is not defined"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58ca373e-fc38-479a-81ed-4f283f675fb5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "marl-env",
   "language": "python",
   "name": "marl-env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
