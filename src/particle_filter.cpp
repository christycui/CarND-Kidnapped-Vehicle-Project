/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 1000;
  particles.resize(num_particles);
  weights.resize(num_particles);

  default_random_engine gen;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // create a normal (Gaussian) distribution
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++) {
    // set weight to 1
    weights[i] = 1;
    // initialize each particle
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
  }
  is_initialized = true;
  cout << "Finish initializing" << endl;
  cout << "particle: " << particles[0].x << " " << particles[0].y << " " << particles[0].weight << endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  default_random_engine gen;

  for (int i = 0; i < num_particles; i++) {
    Particle particle_i = particles[i];
    double theta = particle_i.theta;
    double x = particle_i.x + velocity/yaw_rate*(sin(theta+yaw_rate*delta_t) - sin(theta));
    double y = particle_i.y + velocity/yaw_rate*(cos(theta) - cos(theta+yaw_rate*delta_t));
    double new_theta = theta + yaw_rate*delta_t;
    
    // create a normal (Gaussian) distribution
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(new_theta, std_theta);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
  cout << "Finish prediction" << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++) {
    double x_part = particles[i].x;
    double y_part = particles[i].y;
    double theta = particles[i].theta;
    vector<LandmarkObs> predicted;
    predicted.resize(observations.size());

    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    for (int j = 0; j < observations.size(); j++) {
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;
      // transform to map x coordinate
      double x_map= x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      double y_map= y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
      predicted[j].x = x_map;
      predicted[j].y = y_map;

      x_obs = predicted[j].x;
      y_obs = predicted[j].y;
      double mu_x;
      double mu_y;

      // associations
      double min_distance = sensor_range;
      for (int l = 0; l < map_landmarks.landmark_list.size(); l++) {
        Map::single_landmark_s lm = map_landmarks.landmark_list[l];
        double distance = dist(lm.x_f, lm.y_f, predicted[j].x, predicted[j].y);
        if (distance < min_distance) {
          predicted[j].id = lm.id_i;
          mu_x = lm.x_f;
          mu_y = lm.y_f;
          min_distance = distance;
        }
      }

      associations.push_back(predicted[j].id);
      sense_x.push_back(x_obs);
      sense_y.push_back(y_obs);

      // update weights
      double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));
      double exponent= (pow((x_obs - mu_x), 2))/(2*sig_x*sig_x) + (pow((y_obs - mu_y), 2))/(2*sig_y*sig_y);
      double weight= gauss_norm * exp(-exponent);

      particles[i].weight = particles[i].weight * weight;
    }
    SetAssociations(particles[i], associations, sense_x, sense_y);
    weights[i] = particles[i].weight;
  }

  // normalize weights
  weights = normalize_vector(weights);

  cout << "updated weights" << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  random_device rd;
  mt19937 gen(rd());
  discrete_distribution<> d(weights.begin(), weights.end());
  vector<Particle> particles_new;

  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[d(gen)];
    particles_new.push_back(particle);
    weights[i] = particle.weight;
  }

  // normalize weights
  weights = normalize_vector(weights);
  particles = particles_new;
  cout << "resampled" << endl;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
