import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import LoginImage from "../assets/Login-rafiki.png";
import SignupImage from "../assets/Mobile login-rafiki.png";
// import Page from "./components/page";
import { useNavigate } from "react-router-dom";

import {
    EnvelopeClosedIcon,
    LockClosedIcon,
    PersonIcon,
    GitHubLogoIcon,

} from "@radix-ui/react-icons";





export default function Dashboard() {
    const [isSignIn, setIsSignIn] = useState(true);


    return (
        <div className="h-screen overflow-hidden ">
            <div className="w-full lg:grid lg:min-h-[620px] lg:grid-cols-2 xl:min-h-[800px]">
                {/* Left Column */}
                <div className="border-r-amber-900 flex items-center justify-center py-12 w-full">
                    <div className="mx-auto grid w-[460px] gap-6">
                        {/* Toggle Header */}
                        <div className="grid gap-4 text-center">
                            <div className="space-y-2">
                                {/* <h1 className="text-4xl font-bold tracking-tight">
                                    {isSignIn ? "Welcome back" : "Create an account"}
                                </h1> */}
                                <h2 className="text-3xl font-semibold bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 text-transparent bg-clip-text">
                                    {isSignIn ? "Login" : "Sign Up"}
                                </h2>
                                <p className="text-sm text-muted-foreground">
                                    {isSignIn
                                        ? "Enter your credentials to access your account"
                                        : "Start your journey with us today"}
                                </p>
                            </div>
                            <div className="flex justify-center">
                                <div className="h-1 w-20 bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 rounded-full" />
                            </div>
                        </div>


                        {/* Form */}
                        {isSignIn ? <SignInForm /> : <SignUpForm />}
                        {/* Toggle Link */}
                        <div className="mt-[-5px] text-center text-sm">
                            {isSignIn ? (
                                <>
                                    Don&apos;t have an account?{" "}
                                    <button onClick={() => setIsSignIn(false)} className="underline text-blue-500">
                                        Sign up
                                    </button>
                                </>
                            ) : (
                                <>
                                    Already have an account?{" "}
                                    <button onClick={() => setIsSignIn(true)} className="underline text-blue-500">
                                        Sign in
                                    </button>
                                </>
                            )}
                        </div>
                    </div>
                </div>
                {/* Right Column */}
                <div className="hidden lg:block bg-muted h-full">
                    <img
                        src={isSignIn ? LoginImage : SignupImage}
                        alt="Cover"
                        className="w-full h-full object-cover"
                    />
                </div>
            </div>
        </div>
    );
}

const SignInForm: React.FC = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        email: '',
        password: ''
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        // Add your authentication logic here
        navigate('/dashboard'); // Navigate after successful login
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({
            ...formData,
            [e.target.id]: e.target.value
        });
    };

    return (
        <form className="grid gap-4" onSubmit={handleSubmit}>
            <div className="grid gap-2">
                <Label htmlFor="email">Email</Label>
                <div className="relative">
                    <EnvelopeClosedIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 h-4 w-4" />
                    <Input
                        id="email"
                        type="email"
                        placeholder="m@example.com"
                        className="pl-10"
                        value={formData.email}
                        onChange={handleChange}
                        // required
                    />
                </div>
            </div>
            <div className="grid gap-2">
                <div className="flex items-center justify-between">
                    <Label htmlFor="password">Password</Label>
                    <Link to="/forgot-password" className="text-sm text-blue-500">
                        Forgot password?
                    </Link>
                </div>
                <div className="relative">
                    <LockClosedIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 h-4 w-4" />
                    <Input
                        id="password"
                        type="password"
                        className="pl-10"
                        value={formData.password}
                        onChange={handleChange}
                        // required
                    />
                </div>
            </div>
            <Button type="submit" className="w-full">
                Sign in
            </Button>
            <SocialLoginOptions />
        </form>
    );
};



const SignUpForm: React.FC = () => (
    <form className="grid gap-4 ">
        <div className="flex gap-4">
            <div className="flex-1">
                <Label htmlFor="first-name">First Name</Label>
                <div className="relative">
                    <PersonIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 h-4 w-4" />
                    <Input
                        id="first-name"
                        type="text"
                        placeholder="John"
                        className="pl-10"
                        required
                    />
                </div>
            </div>
            <div className="flex-1">
                <Label htmlFor="last-name">Last Name</Label>
                <div className="relative">
                    <PersonIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 h-4 w-4" />
                    <Input
                        id="last-name"
                        type="text"
                        placeholder="Doe"
                        className="pl-10"
                        required
                    />
                </div>
            </div>
        </div>
        <div className="grid gap-2">
            <Label htmlFor="email">Email</Label>
            <div className="relative">
                <EnvelopeClosedIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 h-4 w-4" />
                <Input
                    id="email"
                    type="email"
                    placeholder="m@example.com"
                    className="pl-10"
                    required
                />
            </div>
        </div>
        <div className="grid gap-2">
            <Label htmlFor="password">Password</Label>
            <div className="relative">
                <LockClosedIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 h-4 w-4" />
                <Input
                    id="password"
                    type="password"
                    className="pl-10"
                    required
                />
            </div>
        </div>
        <div className="grid gap-2">
            <Label htmlFor="confirm-password">Confirm Password</Label>
            <div className="relative">
                <LockClosedIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 h-4 w-4" />
                <Input
                    id="confirm-password"
                    type="password"
                    className="pl-10"
                    required
                />
            </div>
        </div>
        <div className="flex items-center gap-2">
            <Input type="checkbox" className="h-4 w-4" id="terms" required />
            <Label htmlFor="terms" className="text-sm">
                I agree to the{" "}
                <Link to="/terms" className="text-blue-500 hover:underline">
                    terms and conditions
                </Link>
            </Label>
        </div>
        <Button type="submit" className="w-full">
            Create Account
        </Button>
        <SocialLoginOptions />
    </form>
);

const SocialLoginOptions: React.FC = () => (
    <div>
        <div className="relative my-2 ">
            <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
            </div>
            <div className="relative flex justify-center text-sm">
                <span className="bg-background px-2 text-muted-foreground">
                    Or continue with
                </span>
            </div>
        </div>
        <div className="flex flex-row gap-2">
            <Button variant="outline" className="w-full">
                <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24">
                    <path
                        fill="currentColor"
                        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    />
                    <path
                        fill="currentColor"
                        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    />
                    <path
                        fill="currentColor"
                        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                    />
                    <path
                        fill="currentColor"
                        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    />
                </svg>
                Google
            </Button>
            <Button variant="outline" className="w-full">
                <GitHubLogoIcon className="mr-2 h-4 w-4" />
                GitHub
            </Button>
        </div>
    </div>
);